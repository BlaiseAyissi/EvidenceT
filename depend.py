import re

def parse_txt_file(txt_path):
    """Extract package names from requirements.txt-style file."""
    packages = set()
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Extract package name (before any space or version specifier)
            pkg = re.split(r'[<=>\s]', line)[0].strip()
            if pkg:
                packages.add(pkg.lower())
    return packages


def parse_yml_file(yml_path):
    """Extract full package lines and names from environment.yml."""
    packages_full = {}
    with open(yml_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('#', 'channels:', 'dependencies:', '- pip:')):
                continue
            if line.startswith('- '):
                pkg_full = line[2:].strip()
                if pkg_full:
                    pkg_name = re.split(r'[=<>]', pkg_full)[0].strip().lower()
                    packages_full[pkg_name] = pkg_full
    return packages_full


def main(txt_path, yml_path):
    txt_pkgs = parse_txt_file(txt_path)
    yml_pkgs_full = parse_yml_file(yml_path)

    missing = {name: full for name, full in yml_pkgs_full.items() if name not in txt_pkgs}

    if missing:
        print("Packages in YAML but NOT in TXT:\n")
        for full in missing.values():
            print(full)
    else:
        print("✅ All YAML dependencies are present in the TXT file.")

if __name__ == "__main__":
    # Change these paths as needed
    txt_path = "depend.txt"
    yml_path = "environment.yml"
    main(txt_path, yml_path)
