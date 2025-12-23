"""
Explainable AI Visualizer for ML Models
Main Entry Point

A terminal-based application for visualizing and explaining ML model predictions
using LIME, SHAP, and Decision Path explanations.

Design Patterns Implemented:
- Singleton: ModelController
- Factory Method: ExplainerFactory
- Adapter: ModelAdapter
- Strategy: ExplainerBase and implementations
- Observer: ModelSubject and Observer
- Command: CommandManager and SliderChangeCommand
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cli.cli_app import main

if __name__ == "__main__":
    main()
