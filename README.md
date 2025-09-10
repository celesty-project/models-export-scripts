# Models Export Scripts

Python scripts for exporting 3D models of characters and weapons from the String game.

> [!NOTE]  
> The repository is for contributors or people interested in advanced models exporting. If you just want to download the exported models, we kindly host latest versions [here](https://files.celesty.one/Models).

## Prerequisites

Make sure you have the following installed:
- Blender 4.1+
- Python 3.11+

Add blender to your system PATH for easier command line usage.

## Usage

> [!WARNING]  
> You won't be able to use the scripts if you export assets with just FModel alone. The game reuses base meshes for different skins, and FModel does not export configs for materials that are not directly referenced by at least one mesh. Find or write a script that exports all assets and their configs properly.

1. Export all Static/Skeletal Meshes, Materials and Configs (JSON) from the following paths:
    - `PM/Content/PaperMan/SkinAssets/Weapons`;
    - `PM/Content/PaperMan/SkinAssets/Characters`;
    - `PM/Content/PaperMan/Characters`;
    - `PM/Content/PaperMan/Weapons`;
    - `PM/Content/PaperMan/CyWeapons`;

2. Clone the repository:
    ```bash
    git clone https://github.com/celesty-project/models-export-scripts.git
    cd models-export-scripts
    ```

3. Run the export scripts (assuming you have Blender in your PATH):
    ```bash
    # Weapons
    blender -background --python weapons.py -- --path="A:\FModel\Path" --weapons M200 --log-level=ERROR

    # Characters
    blender -background --python characters.py -- --path="A:\FModel\Path" --characters Kanami --log-level=ERROR
    ```

## License

The repo is licensed under the [GNU General Public License v3.0](LICENSE).
