"""
Data Science Persona Package

A streamlined PocketFlow-based data science analysis persona.
Reads repo context and notebook content to provide actionable recommendations.
"""

# Import the main persona
from .persona import DataSciencePersona

# Import PocketFlow classes for convenience
from .pocketflow import Node, Flow, BaseNode

# Import tools
from .file_reader_tool import NotebookReaderTool

__all__ = ["DataSciencePersona", "Node", "Flow", "BaseNode", "NotebookReaderTool"]