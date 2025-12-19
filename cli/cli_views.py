"""
CLIViewManager - View Layer for Terminal Interface
Handles all terminal output formatting and visualization
Implements MVC pattern - this is the View component
"""

import os
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt


class CLIViewManager:
    """
    View layer for CLI application
    
    Responsibilities:
    - Format terminal output
    - Display menus
    - Show status messages
    - Create visualizations (matplotlib popups)
    - Handle user input
    """
    
    def __init__(self):
        """Initialize the view manager"""
        self.width = 60
        
        self.colors = {
            'header': '\033[95m',
            'blue': '\033[94m',
            'cyan': '\033[96m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'bold': '\033[1m',
            'underline': '\033[4m',
            'end': '\033[0m'
        }
    
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def show_welcome(self):
        """Display welcome message"""
        self.clear_screen()
    
    def show_main_menu(self):
        """Display main menu"""
        print("\n" + "=" * self.width)
        print(f"{self.colors['bold']}{self.colors['blue']}▶ MAIN MENU{self.colors['end']}")
        print("=" * self.width)
        print("\n1. Load ML Model")
        print("2. View Model Metadata")
        print("3. Enter Instance for Prediction")
        print("4. Generate Explanation")
        print("5. Undo Last Change (Feature Value)")
        print("6. Redo Last Change")
        print("7. Show Explanation History")
        print("8. Exit")
        print("\n" + "=" * self.width + "\n")
    
    def show_section_header(self, title: str):
        """Display section header"""
        print("\n" + "-" * self.width)
        print(f"{self.colors['bold']}{self.colors['header']}▶ {title}{self.colors['end']}")
        print("-" * self.width + "\n")
    
    def show_success(self, message: str):
        """Display success message"""
        print(f"{self.colors['green']}✓ {message}{self.colors['end']}")
    
    def show_error(self, message: str):
        """Display error message"""
        print(f"{self.colors['red']}✗ Error: {message}{self.colors['end']}")
    
    def show_warning(self, message: str):
        """Display warning message"""
        print(f"{self.colors['yellow']}⚠ {message}{self.colors['end']}")
    
    def show_info(self, message: str):
        """Display info message"""
        print(f"{self.colors['cyan']}ℹ {message}{self.colors['end']}")
    
    def get_input(self, prompt: str) -> str:
        """Get user input with formatted prompt"""
        return input(f"{self.colors['bold']}{prompt}{self.colors['end']}")
    
    def display_explanation(self, explanation: Dict[str, Any], explainer_type: str, feature_names: List[str]):
        """
        Display explanation in terminal and create visualization
        
        Args:
            explanation: Explanation dictionary from explainer
            explainer_type: Type of explainer used
            feature_names: List of feature names
        """
        print("\n" + "=" * self.width)
        print(f"{self.colors['bold']}{self.colors['green']}EXPLANATION RESULTS ({explainer_type.upper()}){self.colors['end']}")
        print("=" * self.width + "\n")
        
        if explainer_type == "lime":
            self._display_lime_explanation(explanation, feature_names)
        elif explainer_type == "shap":
            self._display_shap_explanation(explanation, feature_names)
        elif explainer_type == "decision_path":
            self._display_decision_path_explanation(explanation, feature_names)
        else:
            print(f"  Explanation Type: {explainer_type}")
            print(f"  Data: {explanation}")
        
        print("\n" + "=" * self.width + "\n")
    
    def _display_lime_explanation(self, explanation: Dict[str, Any], feature_names: List[str]):
        """Display LIME explanation with visualization"""
        print("  LIME Feature Importance:\n")
        
        if 'feature_importance' in explanation:
            importance = explanation['feature_importance']
            
            sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
            
            print("  Top Features:")
            for feature, score in sorted_features[:10]:
                bar = self._create_bar(score, max_val=max(abs(s) for _, s in sorted_features))
                sign = "+" if score > 0 else "-"
                print(f"    {feature:30s} {sign} {bar} {abs(score):.4f}")
            
            self._visualize_feature_importance(
                dict(sorted_features[:10]),
                "LIME Feature Importance",
                "Impact on Prediction"
            )
        
        if 'prediction' in explanation:
            print(f"\n  Predicted Class: {explanation['prediction']}")
        
        if 'local_pred' in explanation:
            print(f"  Local Prediction: {explanation['local_pred']:.4f}")
    
    def _display_shap_explanation(self, explanation: Dict[str, Any], feature_names: List[str]):
        """Display SHAP explanation with visualization"""
        print("  SHAP Feature Importance:\n")
        
        if 'shap_values' in explanation:
            shap_values = explanation['shap_values']
            
            importance = {}
            for i, name in enumerate(feature_names):
                if i < len(shap_values):
                    importance[name] = float(shap_values[i])
            
            sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
            
            print("  Top Features:")
            for feature, score in sorted_features[:10]:
                bar = self._create_bar(score, max_val=max(abs(s) for _, s in sorted_features))
                sign = "+" if score > 0 else "-"
                print(f"    {feature:30s} {sign} {bar} {abs(score):.4f}")
            
            self._visualize_feature_importance(
                dict(sorted_features[:10]),
                "SHAP Feature Importance",
                "SHAP Value"
            )
        
        if 'base_value' in explanation:
            print(f"\n  Base Value: {explanation['base_value']:.4f}")
        
        if 'prediction' in explanation:
            print(f"  Predicted Value: {explanation['prediction']:.4f}")
    
    def _display_decision_path_explanation(self, explanation: Dict[str, Any], feature_names: List[str]):
        """Display Decision Path explanation"""
        print("  Decision Path Analysis:\n")
        
        if 'path' in explanation:
            path = explanation['path']
            print(f"  Nodes in Path: {len(path)}")
            print(f"  Path: {path}")
        
        if 'feature_importance' in explanation:
            importance = explanation['feature_importance']
            
            sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
            
            print("\n  Feature Importance in Path:")
            for feature, score in sorted_features[:10]:
                bar = self._create_bar(score, max_val=max(abs(s) for _, s in sorted_features) if sorted_features else 1)
                print(f"    {feature:30s} {bar} {score:.4f}")
            
            self._visualize_feature_importance(
                dict(sorted_features[:10]),
                "Decision Path Feature Importance",
                "Importance Score"
            )
    
    def _create_bar(self, value: float, max_val: float, width: int = 20) -> str:
        """Create ASCII bar chart"""
        if max_val == 0:
            return ""
        
        normalized = abs(value) / max_val
        filled = int(normalized * width)
        empty = width - filled
        
        return "█" * filled + "░" * empty
    
    def _visualize_feature_importance(self, importance: Dict[str, float], title: str, ylabel: str):
        """
        Create matplotlib visualization popup
        
        Args:
            importance: Dictionary of feature names to importance scores
            title: Plot title
            ylabel: Y-axis label
        """
        try:
            features = list(importance.keys())
            values = list(importance.values())
            
            colors = ['#10b981' if v > 0 else '#ef4444' for v in values]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            y_pos = np.arange(len(features))
            ax.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.set_xlabel(ylabel, fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            ax.axvline(x=0, color='black', linewidth=0.8)
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            
            for i, v in enumerate(values):
                ax.text(v, i, f' {v:.3f}', va='center', fontweight='bold', fontsize=9)
            
            plt.tight_layout()
            
            plt.show(block=False)
            plt.pause(0.1)
            
            print(f"\n  {self.colors['cyan']}📊 Visualization opened in new window{self.colors['end']}")
            
        except Exception as e:
            self.show_warning(f"Could not create visualization: {str(e)}")
    
    def visualize_prediction_distribution(self, probabilities: np.ndarray, class_names: List[str] = None):
        """
        Visualize prediction probability distribution
        
        Args:
            probabilities: Array of class probabilities
            class_names: Optional list of class names
        """
        try:
            if class_names is None:
                class_names = [f"Class {i}" for i in range(len(probabilities))]
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(probabilities)))
            bars = ax.bar(class_names, probabilities, color=colors, alpha=0.8, edgecolor='black')
            
            ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
            ax.set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold', pad=20)
            ax.set_ylim([0, 1.0])
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)
            
            print(f"\n  {self.colors['cyan']}📊 Probability distribution opened in new window{self.colors['end']}")
            
        except Exception as e:
            self.show_warning(f"Could not create probability visualization: {str(e)}")
