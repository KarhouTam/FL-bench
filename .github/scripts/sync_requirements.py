#!/usr/bin/env python3
"""
Script to sync requirements.txt with pyproject.toml
"""

import os
import sys
import toml
from packaging.requirements import Requirement

def parse_pyproject_toml(pyproject_path):
    """Parse pyproject.toml and extract dependencies"""
    with open(pyproject_path, 'r') as f:
        pyproject = toml.load(f)
    
    dependencies = []
    
    # Get main dependencies
    if 'tool' in pyproject and 'poetry' in pyproject['tool']:
        poetry_deps = pyproject['tool']['poetry'].get('dependencies', {})
        for package, version in poetry_deps.items():
            if package != 'python':  # Skip python version
                if isinstance(version, dict):
                    # Handle complex version specs
                    if 'version' in version:
                        dependencies.append(f"{package}=={version['version']}" if version['version'].startswith('=') 
                                          else f"{package}{version['version']}")
                else:
                    # Handle simple version specs
                    dependencies.append(f"{package}=={version}" if version.startswith('=') 
                                      else f"{package}{version}")
        
        # Get dev dependencies if they exist
        if 'group' in pyproject['tool']['poetry']:
            for group_name, group in pyproject['tool']['poetry']['group'].items():
                if 'dependencies' in group:
                    for package, version in group['dependencies'].items():
                        if isinstance(version, dict):
                            if 'version' in version:
                                dependencies.append(f"{package}=={version['version']}" if version['version'].startswith('=') 
                                                  else f"{package}{version['version']}")
                        else:
                            dependencies.append(f"{package}=={version}" if version.startswith('=') 
                                              else f"{package}{version}")
    
    return dependencies

def write_requirements_txt(dependencies, requirements_path):
    """Write dependencies to requirements.txt"""
    with open(requirements_path, 'w') as f:
        for dep in sorted(dependencies):
            package, version = dep.split('^')
            f.write(f"{package}=={version}\n")

def main():
    # Define paths
    pyproject_path = os.path.join('.env', 'pyproject.toml')
    requirements_path = os.path.join('.env', 'requirements.txt')
    
    # Check if pyproject.toml exists
    if not os.path.exists(pyproject_path):
        print(f"Error: {pyproject_path} not found")
        sys.exit(1)
    
    # Parse pyproject.toml
    dependencies = parse_pyproject_toml(pyproject_path)
    print(dependencies)
    # Write to requirements.txt
    write_requirements_txt(dependencies, requirements_path)
    
    print(f"Successfully synced {requirements_path} with {pyproject_path}")

if __name__ == "__main__":
    main()