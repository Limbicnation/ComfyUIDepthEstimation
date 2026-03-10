#!/usr/bin/env python3
"""
Validation script for ComfyUI Node Development Skill
"""

import os
import sys
import re
from pathlib import Path


def validate_yaml_frontmatter(content):
    """Validate YAML frontmatter format."""
    pattern = r'^---\n(.*?)\n---'
    match = re.match(pattern, content, re.DOTALL)
    
    if not match:
        return False, "No YAML frontmatter found"
    
    yaml_content = match.group(1)
    required_fields = ['name', 'description']
    
    for field in required_fields:
        if f'{field}:' not in yaml_content:
            return False, f"Missing required field: {field}"
    
    return True, "YAML frontmatter valid"


def validate_word_count(content, max_words=5000):
    """Check word count."""
    words = len(content.split())
    if words > max_words:
        return False, f"Word count too high: {words} (max {max_words})"
    return True, f"Word count: {words}"


def validate_skill(skill_path):
    """Validate a skill directory."""
    skill_path = Path(skill_path)
    
    print(f"🔍 Validating skill: {skill_path.name}")
    print("-" * 50)
    
    # Check required files
    required_files = ['SKILL.md', 'README.md']
    for file in required_files:
        file_path = skill_path / file
        if not file_path.exists():
            print(f"❌ Missing required file: {file}")
            return False
        print(f"✅ Found {file}")
    
    # Validate SKILL.md
    skill_md = skill_path / 'SKILL.md'
    content = skill_md.read_text(encoding='utf-8')
    
    # YAML validation
    valid, msg = validate_yaml_frontmatter(content)
    if valid:
        print(f"✅ {msg}")
    else:
        print(f"❌ {msg}")
        return False
    
    # Word count
    valid, msg = validate_word_count(content)
    if valid:
        print(f"✅ {msg}")
    else:
        print(f"⚠️  {msg}")
    
    print("-" * 50)
    print("✅ Skill validation complete!")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        skill_path = sys.argv[1]
    else:
        skill_path = Path(__file__).parent.parent
    
    success = validate_skill(skill_path)
    sys.exit(0 if success else 1)
