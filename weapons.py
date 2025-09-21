import argparse
import dataclasses
import json
import logging
import pathlib
import re
import sys
import typing

import bpy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FILTERED_PARTS: list[str] = [
    # Audrey's tripod
    "Shelf",
]
"""List of weapon parts to filter out during export."""

FILTERED_WEAPONS: dict[str, str] = {
    "AUG_Rafa": "Rafa was redesigned into Yugiri, all associated weapons were left in game files and are currently broken.",
    "FAMAS_Rafa": "Rafa was redesigned into Yugiri, all associated weapons were left in game files and are currently broken.",
    "M32": "Audrey's ultimate skill weapon. Skipped since its configs are bugged and it is not really needed anyway.",
}
"""Mapping of weapon names to reasons why they
must be filtered out during export.
"""

@dataclasses.dataclass
class Args:
    path: pathlib.Path
    weapons: typing.List[str]
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
class WeaponPartConfig:
    name: str
    mesh: pathlib.Path

@dataclasses.dataclass
class WeaponInfo:
    name: str
    full_name: str
    skin_id: str

@dataclasses.dataclass
class WeaponConfig:
    info: WeaponInfo
    mesh: pathlib.Path
    parts: list[WeaponPartConfig]

    _path: pathlib.Path
    _material_paths: dict[str, pathlib.Path]

    def _parse_texture(self, ue_name: str, ue_path: str) -> Texture | None:
        """Parse a texture from UE4 export data."""
        path = self._path / pathlib.Path(normalize_ue_str(ue_path) + ".png")

        slot = None

        # Case 1: explicit slots
        if ue_name in ("PM_Diffuse", "BaseColor_LOW"):
            slot = "BaseColor"
        elif ue_name in ("N", "PM_Normals"):
            slot = "Normal"
        elif ue_name in ("RM", "PM_SpecularMasks"):
            slot = "RMO"

        # Case 2: detect by asset suffix
        elif ue_name.endswith("_N"):
            slot = "Normal"
        elif ue_name.endswith("_D") or ue_name.endswith("_BaseColor"):
            slot = "BaseColor"
        elif ue_name.endswith("_M") or ue_name.endswith("_Mask"):
            slot = "RMO"  # sometimes mask = packed RMO

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

def get_all_weapons(path: pathlib.Path, export_type: typing.Literal["lobby", "battle"]) -> list[str]:
    """Get all weapons types from meshes configs directory."""
    config_file_name = "Lobby" if export_type == "lobby" else "Mesh"
    
    cy_weapons_path = path / "PM" / "Content" / "PaperMan" / "CyWeapons"
    weapon_types: set[str] = set()

    for f in cy_weapons_path.glob(f"*/Skins/{config_file_name}*.json"):
        weapon_types.add(f.parent.parent.name)

    return sorted(weapon_types)

def get_logging_level(level: str) -> int:
    """Get the logging level from the string."""
    return logging.getLevelNamesMapping().get(level.upper(), logging.INFO)

def filter_weapons(weapons: list[str]) -> list[str]:
    """Filter out FILTERED_WEAPONS."""
    filtered: list[str] = []
    for weapon in weapons:
        reason = FILTERED_WEAPONS.get(weapon)
        if reason:
            logger.warning(
                f"Filtered out weapon {weapon} from exported weapons. Reason: {reason}"
            )
            continue
        filtered.append(weapon)

    return filtered

def parse_args() -> Args:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export weapons from FModel output."
    )

    parser.add_argument(
        "--path",
        type=pathlib.Path,
        default=pathlib.Path(r"C:\FModel\Output\Exports"),
        help="Directory path containing the source exports (default: C:\\FModel\\Output\\Exports)"
    )

    parser.add_argument(
        "--weapons",
        nargs="+",
        default=["all"],
        help="List of weapon names to process (default: all)"
    )

    parser.add_argument(
        "--out-path",
        type=pathlib.Path,
        default=pathlib.Path("./weapons"),
        help="Directory path for exported weapons (default: ./weapons)"
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
        help="Whether to export lobby or battle models of weapons (default: lobby)",
    )

    ns = parser.parse_args()
    
    if "all" in ns.weapons:
        ns.weapons = get_all_weapons(ns.path, ns.export_type)

    ns.weapons = filter_weapons(ns.weapons)
    ns.log_level = get_logging_level(ns.log_level)

    return Args(
        path=ns.path,
        weapons=ns.weapons,
        out_path=ns.out_path,
        log_level=ns.log_level,
        export_type=ns.export_type,
    )

def get_mesh_configs(path: pathlib.Path, weapon_type: str, export_type: typing.Literal["lobby", "battle"]) -> dict[str, dict[str, typing.Any]]:
    """Get the meshes configs for the specified weapon type."""
    config_file_name = "Lobby" if export_type == "lobby" else "Mesh"
    
    configs_folder = path.joinpath("PM", "Content", "PaperMan", "CyWeapons", weapon_type, "Skins")
    configs_files = list(configs_folder.glob(f"{config_file_name}*.json"))

    configs: dict[str, dict[str, typing.Any]] = {}

    for f in configs_files:
        id_str = f.stem.replace(config_file_name, "")
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

def get_weapon_info(full_name: str, skin_id: str, mesh_path: pathlib.Path) -> WeaponInfo:
    """Extract weapon info from the mesh path."""
    # Since weapon mesh path in mesh config always has the same structure:
    # PM/Content/PaperMan/SkinAssets/Weapons/<name>/<base_skin_id>/...
    # We can easily extract weapon name (but not skin_id) from it.
    # Note, that part after base skin id can change, so it is more
    # reliable to use index from the start of the path.
    return WeaponInfo(
        name=mesh_path.parts[5],
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

def get_weapon_config(path: pathlib.Path, full_name: str, skin_id: str, mesh_config: dict[str, typing.Any]) -> WeaponConfig | None:
    """Get a complete weapon config from a mesh config."""
    props = mesh_config.get("Properties")
    
    # Some lobby weapons have "Empty" skins.
    # For example, grenade with its skin 002
    if props is None:
        return None

    mesh_ue_path = props["MainMesh"]["Mesh"]["ObjectPath"]
    mesh_path = ue_mesh_path_to_normal(mesh_ue_path)

    info = get_weapon_info(full_name, skin_id, mesh_path)

    materials_objs = props["MainMesh"].get("Materials")
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
            if key not in material_paths:
                material_paths[key] = path / parse_material_path(obj["Material"])
    
    parts: list[WeaponPartConfig] = []

    for part in props.get("PartSkeletalMesh", []):
        p_name = part["Key"].split("::")[-1]
        p_val = part["Value"]
        
        # Some weapon models can have certain parts missing
        # in lobby but present in battle, and vice versa
        if p_name in FILTERED_PARTS or p_val.get("Mesh") is None:
            continue

        p_mesh_ue_path = p_val["Mesh"]["ObjectPath"]
        p_materials_objs = p_val.get("Materials")

        p_s_material_paths = [
            (path / parse_material_path(obj)) if obj is not None else None
            for obj in p_materials_objs
        ] if p_materials_objs is not None else []

        p_mesh_data = get_ue_mesh_data(path, p_mesh_ue_path)
        p_material_paths: dict[str, pathlib.Path] = {}
        
        for i, obj in enumerate(p_mesh_data["SkeletalMaterials"]):
            key = parse_material_name(obj["Material"])

            if len(p_s_material_paths) >= i + 1 and (p_s_path := p_s_material_paths[i]) is not None:
                p_material_paths[key] = p_s_path
            else:
                if key not in p_material_paths:
                    p_material_paths[key] = path / parse_material_path(obj["Material"])

        material_paths.update(p_material_paths)
        
        cfg = WeaponPartConfig(
            name=p_name,
            mesh=path / ue_mesh_path_to_normal(p_mesh_ue_path)
        )
        parts.append(cfg)

    pendant_data = props.get("PendantSkeletalMesh", [])

    if pendant_part := pendant_data[0] if len(pendant_data) > 0 else None:
        p_mesh_ue_path = pendant_part["Mesh"]["ObjectPath"]
        p_materials_objs = pendant_part.get("Materials")

        p_s_material_paths = [
            (path / parse_material_path(obj)) if obj is not None else None
            for obj in p_materials_objs
        ] if p_materials_objs is not None else []

        p_mesh_data = get_ue_mesh_data(path, p_mesh_ue_path)
        p_material_paths: dict[str, pathlib.Path] = {}
        
        for i, obj in enumerate(p_mesh_data["SkeletalMaterials"]):
            key = parse_material_name(obj["Material"])

            if len(p_s_material_paths) >= i + 1 and (p_s_path := p_s_material_paths[i]) is not None:
                p_material_paths[key] = p_s_path
            else:
                if key not in p_material_paths:
                    p_material_paths[key] = path / parse_material_path(obj["Material"])

        material_paths.update(p_material_paths)
        
        cfg = WeaponPartConfig(
            name="Pendant",
            mesh=path / ue_mesh_path_to_normal(p_mesh_ue_path)
        )
        parts.append(cfg)

    return WeaponConfig(
        info=info,
        mesh=path / mesh_path,
        parts=parts,
        _path=path,
        _material_paths=material_paths,
    )

def clear_scene() -> None:
    """Remove all objects from the current Blender scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

def get_bone_position_from_armature(armature_obj: bpy.types.Object, bone_name: str) -> tuple[float, float, float] | None:
    """Get bone position from armature object."""
    if armature_obj.type != "ARMATURE":
        return None

    armature = typing.cast(bpy.types.Armature, armature_obj.data)
    bone = armature.bones.get(bone_name)
    if bone is None:
        return None

    head_world = armature_obj.matrix_world @ bone.head_local
    return (head_world.x, head_world.y, head_world.z)

def position_weapon_part(part_obj: bpy.types.Object, part_name: str, main_armature: bpy.types.Object) -> bool:
    """Position a weapon part object at the corresponding bone location.
    
    Returns:
        bool: True if positioning was successful, False if bone was not found.
    """
    if not part_obj or not main_armature:
        return False

    position = get_bone_position_from_armature(main_armature, part_name)

    if position is not None:
        part_obj.location = position
        logger.info(f"Positioned {part_name} at bone {part_name}: {position}")
        return True
    else:
        logger.warning(f"Could not find bone {part_name} for part {part_name}")
        return False

def import_weapon_mesh(weapon_config: WeaponConfig) -> tuple[bpy.types.Object | None, bpy.types.Object | None]:
    """Import the main weapon mesh and return mesh object and armature."""
    if not weapon_config.mesh.exists():
        logger.error(f"Main mesh not found: {weapon_config.mesh}")
        return None, None

    bpy.ops.import_scene.gltf(filepath=str(weapon_config.mesh))

    mesh_obj = None
    armature_obj = None
    
    for obj in bpy.context.selected_objects:
        if obj.type == "MESH":
            mesh_obj = obj
        elif obj.type == "ARMATURE":
            armature_obj = obj
    
    if mesh_obj:
        mesh_obj.name = f"{weapon_config.info.name}_{weapon_config.info.skin_id}_MainMesh"
        logger.info(f"Imported main mesh: {mesh_obj.name}")
    
    if armature_obj:
        armature_obj.name = f"{weapon_config.info.name}_{weapon_config.info.skin_id}_Armature"
        logger.info(f"Imported armature: {armature_obj.name}")
    
    return mesh_obj, armature_obj

def import_weapon_part(part_config: WeaponPartConfig, weapon_info: WeaponInfo) -> bpy.types.Object | None:
    """Import a weapon part mesh."""
    if not part_config.mesh.exists():
        logger.error(f"Part mesh not found: {part_config.mesh}")
        return None

    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.import_scene.gltf(filepath=str(part_config.mesh))

    part_obj = next((obj for obj in bpy.context.selected_objects if obj.type == "MESH"), None)
    
    if part_obj:
        part_obj.name = f"{weapon_info.name}_{weapon_info.skin_id}_{part_config.name}"
        logger.info(f"Imported part: {part_obj.name}")
    
    return part_obj

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
        logger.info(f"Applied OpacityScale: {opacity}")
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

def apply_textures_to_object(obj: bpy.types.Object, weapon_config: WeaponConfig) -> None:
    """Apply textures to all materials of an object."""
    data = typing.cast(bpy.types.Mesh, obj.data)

    for material in data.materials:
        if material is None:
            continue
        
        material_name = normalize_ue_str(material.name)
        material_config = weapon_config.get_material_config(material_name)
        if material_config is not None:
            apply_textures_to_existing_material(material, material_config)
        else:
            logger.warning(f"No textures found for material: {material_name}")

def export_weapon(weapon_config: WeaponConfig, output_path: pathlib.Path) -> None:
    """Import all weapon components, position parts, and export as single GLB."""
    logger.info(f"Processing weapon: {weapon_config.info.name} skin {weapon_config.info.skin_id}")
    clear_scene()
    
    main_mesh, armature = import_weapon_mesh(weapon_config)
    if main_mesh is None:
        logger.error("Failed to import main mesh")
        return

    apply_textures_to_object(main_mesh, weapon_config)
    
    all_objects = [main_mesh]
    if armature is not None:
        all_objects.append(armature)

    for part_config in weapon_config.parts:
        part_obj = import_weapon_part(part_config, weapon_config.info)
        if part_obj is not None:

            if armature is not None:
                bone_found = position_weapon_part(part_obj, part_config.name, armature)
                if not bone_found:
                    logger.warning(f"Skipping part {part_config.name} - corresponding bone not found")
                    bpy.data.objects.remove(part_obj, do_unlink=True)
                    continue

            apply_textures_to_object(part_obj, weapon_config)
            all_objects.append(part_obj)

    bpy.ops.object.select_all(action='DESELECT')
    for obj in all_objects:
        obj.select_set(True)

    if all_objects and bpy.context.view_layer is not None:
        bpy.context.view_layer.objects.active = all_objects[0]
    else:
        logger.error("No objects to set as active for export")

    output_file = output_path / weapon_config.info.full_name / f"{weapon_config.info.skin_id}.glb"
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

    logger.info(f"Exported weapon: {output_file}")

def process_weapon(weapon_name: str, args: Args) -> None:
    """Process all skins for a given weapon."""
    logger.info(f"Processing weapon: {weapon_name}")
    
    try:
        meshes_configs = get_mesh_configs(args.path, weapon_name, args.export_type)
        logger.info(f"Found {len(meshes_configs)} skins for {weapon_name}")

        for skin_id, mesh_config in meshes_configs.items():
            try:
                weapon_config = get_weapon_config(args.path, weapon_name, skin_id, mesh_config)
                if weapon_config is None:
                    continue
                export_weapon(weapon_config, args.out_path)
            except Exception as e:
                exc_info = (type(e), e, e.__traceback__)
                logger.error(f"Fatal error during processing skin {skin_id} of {weapon_name}: {e}", exc_info=exc_info)

    except Exception as e:
        exc_info = (type(e), e, e.__traceback__)
        logger.error(f"Fatal error during processing {weapon_name}: {e}", exc_info=exc_info)

def main() -> None:
    """Main entry point."""
    args = parse_args()
    logger.setLevel(args.log_level)
    args.out_path.mkdir(parents=True, exist_ok=True)

    logger.info("The following weapons will be exported: %s", ", ".join(args.weapons))

    for weapon in args.weapons:
        process_weapon(weapon, args)
    
    logger.info("Processing complete")

if __name__ == "__main__":
    # When running inside Blender, pass arguments after "--"
    if "--" in sys.argv:
        idx = sys.argv.index("--") + 1
        main_args = sys.argv[idx:]
        sys.argv = [sys.argv[0]] + main_args
    main()
