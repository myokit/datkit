#
# Datkit's version info
#
# This file is part of Datkit.
# See http://myokit.org for copyright, sharing, and licensing details.
#

# Version as a tuple (major, minor, revision)
#  - Changes to major are rare
#  - Changes to minor indicate new features, possible slight backwards
#    incompatibility
#  - Changes to revision indicate bugfixes, tiny new features
#  - There is no significance to odd/even numbers
__version_tuple__ = 0, 0, 1

# String version of the version number
__version__ = '.'.join([str(x) for x in __version_tuple__])

