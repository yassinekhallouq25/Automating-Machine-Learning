Title: Replace setup.py with pyproject.toml for pricing_checks package

Description:

The pricing_checks package was originally configured using setup.py. This worked, but setup.py is no longer the preferred standard for modern Python packaging.

This issue tracks the update to move the package configuration from setup.py to pyproject.toml, which is now the standard approach for defining Python package metadata and build settings.

Changes made:

Removed the old setup.py packaging configuration.
Added a pyproject.toml file for the pricing_checks package.
Moved package metadata, dependencies, and build configuration into pyproject.toml.
Updated the package structure to follow modern Python packaging standards.

Reason for change:

Using pyproject.toml makes the package easier to maintain and aligns it with current Python packaging best practices.

Acceptance criteria:

The package can be installed successfully.
The package can be built using the new pyproject.toml configuration.
No packaging information remains duplicated between setup.py and pyproject.toml.
Existing functionality of pricing_checks continues to work after the packaging update.
