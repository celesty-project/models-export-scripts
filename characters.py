import argparse
import dataclasses
import json
import logging
import pathlib
import re
import sys
import typing

import bpy
import bmesh

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FILTERED_CHARACTERS: dict[str, str] = {
    "": ""
}
"""Mapping of character names to reasons why they
must be filtered out during export.
"""

@dataclasses.dataclass
class Args:
    path: pathlib.Path
    characters: typing.List[str]
    out_path: pathlib.Path
    log_level: int
    export_type: typing.Literal["lobby", "battle"]

@dataclasses.dataclass
class Texture:
    slot: str
    path: pathlib.Path

@dataclasses.dataclass
class MaterialConfig:
    textures: list[Texture]
    params: dict[str, typing.Any]

@dataclasses.dataclass
class CharacterInfo:
    name: str
    full_name: str
    skin_id: str

@dataclasses.dataclass
class CharacterConfig:
    info: CharacterInfo
    mesh: pathlib.Path

    _path: pathlib.Path
    _material_paths: dict[str, pathlib.Path]
    
    def get_path(self) -> pathlib.Path:
        """Get the base path."""
        return self._path

    def _parse_texture(self, ue_name: str, ue_path: str) -> Texture | None:
        """Parse a texture from UE4 export data."""
        path = self._path / pathlib.Path(normalize_ue_str(ue_path) + ".png")

        slot = None

        # Case 1: explicit slots
        if ue_name in ("PM_Diffuse", "BaseColor_LOW", "BaseMap"):
            slot = "BaseColor"
        elif ue_name in ("N", "PM_Normals"):
            slot = "Normal"

        # Case 2: detect by asset suffix
        elif ue_name.endswith("_N"):
            slot = "Normal"
        elif ue_name.endswith("_D") or ue_name.endswith("_BaseColor"):
            slot = "BaseColor"

        if not slot:
            logger.debug(f"Unknown texture slot: {ue_name}. Skipping.")
            return None

        return Texture(slot=slot, path=path)

    def get_material_config(self, material_name: str) -> MaterialConfig | None:
        """Get the textures for the specified material of the skin."""
        material_config_path = self._material_paths.get(material_name)

        if material_config_path is None or not material_config_path.exists():
            logger.error(f"Material config not found: {material_name}. Existing configs: {list(self._material_paths.keys())}")
            return None

        material_config = json.load(open(material_config_path))

        textures = [
            self._parse_texture(ue_name, ue_path)
            for ue_name, ue_path in material_config.get("Textures", {}).items()
        ]
        filtered_textures = filter_none(textures)

        return MaterialConfig(textures=filtered_textures, params=material_config.get("Parameters", {}))

@dataclasses.dataclass
class MaterialActionRemove:
    pass

@dataclasses.dataclass
class MaterialActionApplyCfg:
    _textures: list[Texture]
    _params: dict[str, typing.Any]
    
    def get_config(self, path: pathlib.Path) -> MaterialConfig:
        textures = [
            Texture(
                slot=texture.slot,
                path=path / texture.path.with_suffix(".png")
            )
            for texture in self._textures
        ]
        return MaterialConfig(textures=textures, params=self._params)

@dataclasses.dataclass
class SpecialMaterialConfig:
    action: MaterialActionRemove | MaterialActionApplyCfg

SPECIAL_MATERIALS: dict[str, SpecialMaterialConfig] = {
    "MI_Flavia_FX_01": SpecialMaterialConfig(
        action=MaterialActionRemove()
    ),
    "MI_Flavia_S206_dress": SpecialMaterialConfig(
        action=MaterialActionApplyCfg(
            _textures=[
                Texture(
                    slot="BaseColor",
                    path=pathlib.Path("PM/Content/PaperMan/SkinAssets/Characters/Flavia/S206/Mesh3D/Textures/T_Flavia_Body1_S206_D")
                ),
            ],
            _params={}
        ),
    ),
}
"""Mapping of material names special handling configs."""

def get_all_characters(path: pathlib.Path, export_type: typing.Literal["lobby", "battle"]) -> list[str]:
    """Get all character types from meshes configs directory."""
    config_file_name = "Lobby" if export_type == "lobby" else "Mesh"

    characters_path = path / "PM" / "Content" / "PaperMan" / "Characters"
    character_types: set[str] = set()

    if export_type == "lobby":
        glob = f"*/Lobby/Skins/{config_file_name}*.json"
    else:
        glob = f"*/Skins/{config_file_name}*.json"

    for f in characters_path.glob(glob):
        parent_p = f.parent.parent
        character_types.add(parent_p.parent.name if export_type == "lobby" else parent_p.name)

    return sorted(character_types)

def get_logging_level(level: str) -> int:
    """Get the logging level from the string."""
    return logging.getLevelNamesMapping().get(level.upper(), logging.INFO)

def filter_characters(characters: list[str]) -> list[str]:
    """Filter out FILTERED_CHARACTERS."""
    filtered: list[str] = []
    for character in characters:
        reason = FILTERED_CHARACTERS.get(character)
        if reason:
            logger.warning(
                f"Filtered out character {character} from exported characters. Reason: {reason}"
            )
            continue
        filtered.append(character)

    return filtered

def parse_args() -> Args:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export characters from FModel output."
    )

    parser.add_argument(
        "--path",
        type=pathlib.Path,
        default=pathlib.Path(r"C:\FModel\Output\Exports"),
        help="Directory path containing the source exports (default: C:\\FModel\\Output\\Exports)"
    )

    parser.add_argument(
        "--characters",
        nargs="+",
        default=["all"],
        help="List of character names to process (default: all)"
    )

    parser.add_argument(
        "--out-path",
        type=pathlib.Path,
        default=pathlib.Path("./characters"),
        help="Directory path for exported characters (default: ./characters)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--export-type",
        choices=["lobby", "battle"],
        default="lobby",
        help="Whether to export lobby or battle models of characters (default: lobby)",
    )

    ns = parser.parse_args()
    
    if "all" in ns.characters:
        ns.characters = get_all_characters(ns.path, ns.export_type)

    ns.characters = filter_characters(ns.characters)
    ns.log_level = get_logging_level(ns.log_level)

    return Args(
        path=ns.path,
        characters=ns.characters,
        out_path=ns.out_path,
        log_level=ns.log_level,
        export_type=ns.export_type,
    )

def get_mesh_configs(path: pathlib.Path, character_type: str, export_type: typing.Literal["lobby", "battle"]) -> dict[str, dict[str, typing.Any]]:
    """Get the meshes configs for the specified character type."""
    config_file_name = "Lobby" if export_type == "lobby" else "Mesh"
    
    if export_type == "lobby":
        configs_folder = path.joinpath("PM", "Content", "PaperMan", "Characters", character_type, "Lobby", "Skins")
    else:
        configs_folder = path.joinpath("PM", "Content", "PaperMan", "Characters", character_type, "Skins")

    configs: dict[str, dict[str, typing.Any]] = {}
    pattern = re.compile(rf"^{config_file_name}(\d+)$")

    for f in configs_folder.glob("*.json"):
        match = pattern.match(f.stem)
        if match is None:
            continue

        id_str = match.group(1)
        data = json.load(open(f))

        # FModel Desktop exports JSON files as lists, for some reason
        configs[id_str] = data[0] if isinstance(data, list) else data

    return configs

def normalize_ue_str(ue_path: str) -> str:
    """Normalize a UE4 asset path by removing object suffix."""
    return re.sub(r"\.(\d+|[^/.]+)$", "", ue_path)

def ue_mesh_path_to_normal(ue_mesh_path: str) -> pathlib.Path:
    """Convert a UE mesh path to a normal file path."""
    return pathlib.Path(normalize_ue_str(ue_mesh_path) + ".glb")

def get_ue_mesh_data(path: pathlib.Path, ue_mesh_path: str) -> dict[str, typing.Any]:
    """Get the UE mesh data for the specified mesh path."""
    file_path = path / pathlib.Path(normalize_ue_str(ue_mesh_path) + ".json")
    return json.load(open(file_path))

def get_character_info(full_name: str, skin_id: str, mesh_path: pathlib.Path) -> CharacterInfo:
    """Extract character info from the mesh path."""
    # Since character mesh path in mesh config always has the same structure:
    # PM/Content/PaperMan/SkinAssets/Characters/<name>/<base_skin_id>/...
    # We can easily extract character name (but not skin_id) from it.
    # Note, that part after base skin id can change, so it is more
    # reliable to use index from the start of the path.
    return CharacterInfo(
        name=mesh_path.parts[6],
        full_name=full_name,
        skin_id=skin_id,
    )

def parse_material_name(material_obj: dict[str, typing.Any]) -> str:
    """Parse UE material name.

    Example name: MaterialInstanceConstant'MI_Weap_M82_01_204'
    Parsed: MI_Weap_M82_01_204
    """
    return material_obj["ObjectName"].split("'")[1]

def parse_material_path(material_obj: dict[str, typing.Any]) -> pathlib.Path:
    """Parse UE material path."""
    return pathlib.Path(normalize_ue_str(material_obj["ObjectPath"]) + ".json")

def filter_none(lst: list[typing.Any]) -> list[typing.Any]:
    """Filter out None values from a list."""
    return [item for item in lst if item is not None]

def get_character_config(path: pathlib.Path, full_name: str, skin_id: str, mesh_config: dict[str, typing.Any]) -> CharacterConfig | None:
    """Get a complete character config from a mesh config."""
    mesh_ue_path = mesh_config["Properties"]["MainMesh"]["Mesh"]["ObjectPath"]
    mesh_path = ue_mesh_path_to_normal(mesh_ue_path)

    info = get_character_info(full_name, skin_id, mesh_path)

    materials_objs = mesh_config["Properties"]["MainMesh"].get("Materials")
    s_material_paths = [
        (path / parse_material_path(obj)) if obj is not None else None
        for obj in materials_objs
    ] if materials_objs is not None else []

    mesh_data = get_ue_mesh_data(path, mesh_ue_path)
    material_paths: dict[str, pathlib.Path] = {}
    
    for i, obj in enumerate(mesh_data["SkeletalMaterials"]):
        key = parse_material_name(obj["Material"])

        if len(s_material_paths) >= i + 1 and (s_path := s_material_paths[i]) is not None:
            material_paths[key] = s_path
        else:
            material_paths[key] = path / parse_material_path(obj["Material"])

    return CharacterConfig(
        info=info,
        mesh=path / mesh_path,
        _path=path,
        _material_paths=material_paths,
    )

def clear_scene() -> None:
    """Remove all objects from the current Blender scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

def import_character_mesh(character_config: CharacterConfig) -> tuple[bpy.types.Object | None, bpy.types.Object | None]:
    """Import the main character mesh and return mesh object and armature."""
    if not character_config.mesh.exists():
        logger.error(f"Main mesh not found: {character_config.mesh}")
        return None, None

    bpy.ops.import_scene.gltf(filepath=str(character_config.mesh))

    mesh_obj = None
    armature_obj = None
    
    for obj in bpy.context.selected_objects:
        if obj.type == "MESH":
            mesh_obj = obj
        elif obj.type == "ARMATURE":
            armature_obj = obj
    
    if mesh_obj:
        mesh_obj.name = f"{character_config.info.name}_{character_config.info.skin_id}_MainMesh"
        logger.info(f"Imported main mesh: {mesh_obj.name}")
    
    if armature_obj:
        armature_obj.name = f"{character_config.info.name}_{character_config.info.skin_id}_Armature"
        logger.info(f"Imported armature: {armature_obj.name}")
    
    return mesh_obj, armature_obj

def ensure_principled_bsdf(nodes: bpy.types.Nodes, links: bpy.types.NodeLinks) -> bpy.types.ShaderNodeBsdfPrincipled:
    """Find or create a Principled BSDF node and connect it to output if missing."""
    bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if bsdf:
        return typing.cast(bpy.types.ShaderNodeBsdfPrincipled, bsdf)

    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)

    output = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
    if output:
        links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    return typing.cast(bpy.types.ShaderNodeBsdfPrincipled, bsdf)

def load_image_node(nodes: bpy.types.Nodes, path: pathlib.Path, y_position: float) -> bpy.types.ShaderNodeTexImage | None:
    """Load an image from path into a new Texture node."""
    if not path.exists():
        logger.warning(f"Texture not found: {path}")
        return None

    try:
        node = typing.cast(bpy.types.ShaderNodeTexImage, nodes.new(type="ShaderNodeTexImage"))
        node.location = (-300, y_position)
        node.image = bpy.data.images.load(str(path))
        return node
    except Exception as e:
        logger.warning(f"Failed to load image {path}: {e}")
        return None

def apply_base_color(
    links: bpy.types.NodeLinks,
    bsdf: bpy.types.ShaderNodeBsdfPrincipled,
    node: bpy.types.ShaderNodeTexImage,
    is_transparent: bool = False
) -> None:
    """Connect the base color texture to the BSDF shader."""
    base_color_input = bsdf.inputs["Base Color"]
    if base_color_input.links is not None:
        for link in list(base_color_input.links):
            links.remove(link)
    
    links.new(node.outputs["Color"], base_color_input)
    
    if is_transparent:
        alpha_input = bsdf.inputs["Alpha"]
        if alpha_input.links is not None:
            for link in list(alpha_input.links):
                links.remove(link)

        links.new(node.outputs["Alpha"], alpha_input)

def apply_normal_map(
    nodes: bpy.types.Nodes,
    links: bpy.types.NodeLinks,
    bsdf: bpy.types.ShaderNodeBsdfPrincipled,
    node: bpy.types.ShaderNodeTexImage,
    y_pos: float
) -> None:
    """Connect the normal map texture to the BSDF shader."""
    normal_map = nodes.new(type="ShaderNodeNormalMap")
    normal_map.location = (-100, y_pos)
    links.new(node.outputs["Color"], normal_map.inputs["Color"])
    links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])

def apply_rmo(
    nodes: bpy.types.Nodes,
    links: bpy.types.NodeLinks,
    bsdf: bpy.types.ShaderNodeBsdfPrincipled,
    node: bpy.types.ShaderNodeTexImage,
    y_pos: float
) -> None:
    """Connect the roughness and metallic maps to the BSDF shader."""
    sep = nodes.new(type="ShaderNodeSeparateRGB")
    sep.location = (-100, y_pos)
    links.new(node.outputs["Color"], sep.inputs["Image"])
    links.new(sep.outputs["G"], bsdf.inputs["Roughness"])
    links.new(sep.outputs["B"], bsdf.inputs["Metallic"])

def apply_material_parameters(bsdf: bpy.types.ShaderNodeBsdfPrincipled, config: MaterialConfig) -> None:
    """Apply scalar and color parameters from material config to the BSDF shader."""
    params = config.params
    scalars = params.get("Scalars", {})
    
    if "MetallicOffset" in scalars:
        typing.cast(
            bpy.types.NodeSocketFloat,
            bsdf.inputs["Metallic"]
        ).default_value = float(scalars["MetallicOffset"])
    if "RoughnessOffset" in scalars:
        typing.cast(
            bpy.types.NodeSocketFloat,
            bsdf.inputs["Roughness"]
        ).default_value = float(scalars["RoughnessOffset"])
    if "Roughness" in scalars:
        typing.cast(
            bpy.types.NodeSocketFloat,
            bsdf.inputs["Roughness"]
        ).default_value = float(scalars["Roughness"])

    if "OpacityScale" in scalars:
        opacity = float(scalars["OpacityScale"])
        typing.cast(bpy.types.NodeSocketFloat, bsdf.inputs["Alpha"]).default_value = opacity
        logger.info(f"Applied opacity: {opacity}")
    elif "Opacity" in scalars:
        opacity = float(scalars["Opacity"])
        typing.cast(bpy.types.NodeSocketFloat, bsdf.inputs["Alpha"]).default_value = opacity
        logger.info(f"Applied Opacity: {opacity}")
    
    colors = params.get("Colors", {})
    
    # Base color tint (for materials without diffuse textures)
    if "BaseColorTint" in colors:
        tint = colors["BaseColorTint"]
        base_color_input = typing.cast(bpy.types.NodeSocketColor, bsdf.inputs["Base Color"])
        base_color_input.default_value = (tint["R"], tint["G"], tint["B"], 1.0)

        logger.info(f"Applied BaseColorTint: R={tint['R']:.3f}, G={tint['G']:.3f}, B={tint['B']:.3f}")

def handle_unlit_material(material: bpy.types.Material, bsdf: bpy.types.ShaderNodeBsdfPrincipled, config: MaterialConfig) -> None:
    """Handle unlit materials by setting up emission properties."""
    params = config.params
    properties = params.get("Properties", {})
    base_overrides = properties.get("BasePropertyOverrides", {})
    scalars = params.get("Scalars", {})

    is_unlit = base_overrides.get("ShadingModel") == "MSM_Unlit"
    
    if is_unlit:
        logger.info(f"Material {material.name} is unlit, setting up emission")

        if material.node_tree is None:
            logger.error(f"Material {material.name} has no node tree for unlit setup")
            return

        nodes = material.node_tree.nodes
        links = material.node_tree.links

        emission_shader = nodes.new(type="ShaderNodeEmission")
        emission_shader.location = (100, 0)
    
        base_color = typing.cast(bpy.types.NodeSocketColor, bsdf.inputs["Base Color"]).default_value

        base_color_input = bsdf.inputs["Base Color"]
        base_color_texture_node = None
        if base_color_input.links:
            connected_node = base_color_input.links[0].from_node
            if connected_node and connected_node.type == 'TEX_IMAGE':
                base_color_texture_node = connected_node
        
        typing.cast(bpy.types.NodeSocketColor, emission_shader.inputs["Color"]).default_value = base_color

        emission_scale = scalars.get("Emissive Scale", 1.0)
        typing.cast(
            bpy.types.NodeSocketFloat,
            emission_shader.inputs["Strength"]
        ).default_value = float(emission_scale)

        output_node = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
        if output_node:
            surface_links = output_node.inputs["Surface"].links
            if surface_links:
                for link in surface_links:
                    links.remove(link)

            is_translucent = params.get("IsTranslucent", False) or base_overrides.get("BlendMode") == "BLEND_Translucent"
            if is_translucent:
                material.surface_render_method = 'BLENDED'

                # Create a Principled BSDF for transparency and set it to emission mode
                translucent_bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
                translucent_bsdf.location = (100, -100)

                base_color_input_ = typing.cast(bpy.types.NodeSocketColor, translucent_bsdf.inputs["Base Color"])
                base_color_input_.default_value = base_color

                if base_color_texture_node:
                    links.new(base_color_texture_node.outputs["Color"], translucent_bsdf.inputs["Base Color"])
                    alpha_input = translucent_bsdf.inputs["Alpha"]
                    if alpha_input.links:
                        for link in list(alpha_input.links):
                            links.remove(link)
                            
                    links.new(base_color_texture_node.outputs["Alpha"], alpha_input)
                    logger.info(f"Connected base color texture (with alpha) to translucent unlit material: {material.name}")
                
                emission_scale = float(scalars.get("Emissive Scale", 1.0))
                if "Emission Color" in translucent_bsdf.inputs:
                    if base_color_texture_node:
                        links.new(base_color_texture_node.outputs["Color"], translucent_bsdf.inputs["Emission Color"])
                    else:
                        emission_color = typing.cast(bpy.types.NodeSocketColor, translucent_bsdf.inputs["Emission Color"])
                        emission_color.default_value = base_color
                if "Emission Strength" in translucent_bsdf.inputs:
                    emission_strength_input = typing.cast(
                        bpy.types.NodeSocketFloat,
                        translucent_bsdf.inputs["Emission Strength"]
                    )
                    emission_strength_input.default_value = emission_scale
                elif "Emission" in translucent_bsdf.inputs:
                    # Older Blender versions
                    emission_input = typing.cast(bpy.types.NodeSocketFloat, translucent_bsdf.inputs["Emission"])
                    emission_input.default_value = emission_scale

                opacity_scale = scalars.get("OpacityScale", 1.0)
                opacity = 1.0 if is_translucent and opacity_scale > 1.0 else min(opacity_scale, 1.0)
                
                alpha = typing.cast(bpy.types.NodeSocketFloat, translucent_bsdf.inputs["Alpha"])
                alpha.default_value = float(opacity)

                links.new(translucent_bsdf.outputs["BSDF"], output_node.inputs["Surface"])
                
                logger.info(f"Applied translucent emission with opacity: {opacity}")
            else:
                # Non-translucent: connect emission directly to output
                # If there's a base color texture, connect it to emission
                if base_color_texture_node:
                    links.new(base_color_texture_node.outputs["Color"], emission_shader.inputs["Color"])
                    logger.info(f"Connected base color texture to emission shader: {material.name}")
                
                links.new(emission_shader.outputs["Emission"], output_node.inputs["Surface"])

def apply_textures_to_existing_material(material: bpy.types.Material, config: MaterialConfig) -> None:
    """Apply textures to an existing material by modifying its node tree."""
    if not material.use_nodes:
        material.use_nodes = True

    if material.node_tree is None:
        logger.error(f"Material {material.name} has no node tree.")
        return

    nodes = material.node_tree.nodes
    links = material.node_tree.links

    bsdf = ensure_principled_bsdf(nodes, links)

    texture_by_slot: dict[str, "Texture"] = {t.slot: t for t in config.textures if t.slot not in {}}
    has_base_color_texture = "BaseColor" in texture_by_slot

    apply_material_parameters(bsdf, config)

    is_transparent = (
        config.params.get("IsTranslucent", False) or 
        config.params.get("Properties", {}).get("BasePropertyOverrides", {}).get("BlendMode") == "BLEND_Translucent"
    )
    
    is_translucent_unlit = (
        is_transparent and 
        config.params.get("Properties", {}).get("BasePropertyOverrides", {}).get("ShadingModel") == "MSM_Unlit"
    )

    y_offset = 0
    texture_count = 0

    for slot, texture in texture_by_slot.items():
        node = load_image_node(nodes, texture.path, y_offset)
        if not node:
            continue

        if slot == "BaseColor":
            apply_base_color(links, bsdf, node, is_transparent)
            logger.info(f"Applied diffuse texture for {material.name}, overriding BaseColorTint" + 
                       (" (with alpha channel)" if is_transparent else ""))
        elif slot == "Normal":
            apply_normal_map(nodes, links, bsdf, node, y_offset)
        elif slot == "RMO":
            apply_rmo(nodes, links, bsdf, node, y_offset)
        else:
            logger.debug(f"Unhandled texture slot: {slot}")
            continue

        y_offset -= 250
        texture_count += 1

    if not has_base_color_texture or is_translucent_unlit:
        colors = config.params.get("Colors", {})
        if "BaseColorTint" in colors:
            logger.info(f"No diffuse texture found for {material.name}, using BaseColorTint only")

    base_overrides = config.params.get("Properties", {}).get("BasePropertyOverrides", {})
    material.use_backface_culling = not base_overrides.get("TwoSided", False)

    # Handle unlit materials (emission-based)
    handle_unlit_material(material, bsdf, config)

    logger.info(f"Applied {texture_count} textures and material parameters to '{material.name}'")

def remove_faces_with_material(obj: bpy.types.Object, mat_index: int) -> None:
    """Remove all faces that use the given material index."""
    mesh = typing.cast(bpy.types.Mesh, obj.data)
    bm = bmesh.new()
    bm.from_mesh(mesh)

    faces_to_remove = [f for f in bm.faces if f.material_index == mat_index]
    bmesh.ops.delete(bm, geom=faces_to_remove, context="FACES")

    bm.to_mesh(mesh)
    bm.free()

    if mat_index < len(mesh.materials):
        mesh.materials.pop(index=mat_index)

def apply_textures_to_object(obj: bpy.types.Object, character_config: CharacterConfig) -> None:
    """Apply textures to all materials of an object."""
    data = typing.cast(bpy.types.Mesh, obj.data)
        
    for i, material in enumerate(list(data.materials)):
        if material is None:
            continue

        material_name = normalize_ue_str(material.name)
        special_material = SPECIAL_MATERIALS.get(material_name)

        if special_material is not None:
            if isinstance(special_material.action, MaterialActionRemove):
                logger.info(f"Removing special material and geometry: {material_name}")
                remove_faces_with_material(obj, i)
                continue
            else:
                material_config = special_material.action.get_config(character_config.get_path())
                apply_textures_to_existing_material(material, material_config)
            continue

        material_config = character_config.get_material_config(material_name)
        if material_config is not None:
            apply_textures_to_existing_material(material, material_config)
        else:
            logger.warning(f"No textures found for material: {material_name}")

def export_character(character_config: CharacterConfig, output_path: pathlib.Path) -> None:
    """Import all character components, textures, and export as single GLB."""
    logger.info(f"Processing character: {character_config.info.name} skin {character_config.info.skin_id}")
    clear_scene()

    main_mesh, armature = import_character_mesh(character_config)
    if main_mesh is None:
        logger.error("Failed to import main mesh")
        return

    apply_textures_to_object(main_mesh, character_config)
    
    all_objects = [main_mesh]
    if armature is not None:
        all_objects.append(armature)

    bpy.ops.object.select_all(action='DESELECT')
    for obj in all_objects:
        obj.select_set(True)

    if all_objects and bpy.context.view_layer is not None:
        bpy.context.view_layer.objects.active = all_objects[0]
    else:
        logger.error("No objects to set as active for export")

    output_file = output_path / character_config.info.full_name / f"{character_config.info.skin_id}.glb"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.export_scene.gltf(
        filepath=str(output_file),
        use_selection=True,
        export_format='GLB',
        # Some skins have vertex colors baked into the model that Blender mixes
        # with the Base Color texture by default, causing darkened textures.
        # Setting export_vertex_color='NONE' disables exporting vertex colors,
        # which prevents unwanted mixing and ensures textures appear correctly.
        export_vertex_color='NONE',
    )

    logger.info(f"Exported character: {output_file}")

def process_character(character_name: str, args: Args) -> None:
    """Process all skins for a given character."""
    logger.info(f"Processing character: {character_name}")

    try:
        mesh_configs = get_mesh_configs(args.path, character_name, args.export_type)
        logger.info(f"Found {len(mesh_configs)} skins for {character_name}")

        for skin_id, mesh_config in mesh_configs.items():
            try:
                character_config = get_character_config(args.path, character_name, skin_id, mesh_config)
                if character_config is None:
                    continue
                export_character(character_config, args.out_path)
            except Exception as e:
                exc_info = (type(e), e, e.__traceback__)
                logger.error(f"Fatal error during processing skin {skin_id} of {character_name}: {e}", exc_info=exc_info)

    except Exception as e:
        exc_info = (type(e), e, e.__traceback__)
        logger.error(f"Fatal error during processing {character_name}: {e}", exc_info=exc_info)

def main() -> None:
    """Main entry point."""
    args = parse_args()
    logger.setLevel(args.log_level)
    args.out_path.mkdir(parents=True, exist_ok=True)

    logger.info("The following characters will be exported: %s", ", ".join(args.characters))

    for character in args.characters:
        process_character(character, args)
    
    logger.info("Processing complete")

if __name__ == "__main__":
    # When running inside Blender, pass arguments after "--"
    if "--" in sys.argv:
        idx = sys.argv.index("--") + 1
        main_args = sys.argv[idx:]
        sys.argv = [sys.argv[0]] + main_args
    main()
