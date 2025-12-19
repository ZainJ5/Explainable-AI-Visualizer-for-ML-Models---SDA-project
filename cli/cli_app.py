"""
ExplainableAICLI - Terminal-based Interface
Main CLI Application implementing MVC pattern and Observer pattern
"""

import os
import sys
from typing import List, Dict, Any
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domain.model_controller import ModelController
from domain.explainer_factory import ExplainerFactory
from event.observer import Observer
from event.model_subject import ModelSubject
from command.command_manager import CommandManager
from command.slider_change_command import SliderChangeCommand
from cli.cli_views import CLIViewManager


class ExplainableAICLI(Observer):
    """
    Main CLI Application implementing Observer pattern
    
    This class serves as the main CLI controller and observes model changes
    through the Observer pattern implementation.
    
    Design Patterns:
    - Observer: Observes model state changes
    - MVC: This is the Controller, CLIViewManager is the View, domain classes are Model
    - Command: Uses CommandManager for undo/redo
    - Singleton: Uses ModelController singleton
    - Factory: Uses ExplainerFactory
    """
    
    def __init__(self):
        """Initialize the CLI application"""
        self.model_controller = ModelController()  
        self.model_subject = ModelSubject() 
        self.command_manager = CommandManager()  
        self.view_manager = CLIViewManager()  
        self.model_subject.attach(self)
        
        self.current_instance = []
        self.current_explainer = None
        self.explainer_type = "auto"
        self.explanation_history = []
        
    def update(self, data: Dict[str, Any]):
        """
        Observer pattern update method
        
        Args:
            data: Dictionary containing event type and associated data
        """
        event_type = data.get('type', 'unknown')
        
        if event_type == "model_loaded":
            feature_names = data.get('feature_names', [])
            self.view_manager.show_info(f"Model loaded with {len(feature_names)} features")
        elif event_type == "prediction":
            prediction = data.get('prediction')
            self.view_manager.show_info(f"Prediction made: {prediction}")
        elif event_type == "instance_changed":
            self.view_manager.show_info("Instance values updated")
    
    def run(self):
        """Main application loop"""
        while True:
            try:
                self.view_manager.show_main_menu()
                choice = self.view_manager.get_input("Enter your choice (1-8): ")
                
                if choice == "1":
                    self.load_model()
                elif choice == "2":
                    self.view_model_metadata()
                elif choice == "3":
                    self.enter_instance()
                elif choice == "4":
                    self.generate_explanation()
                elif choice == "5":
                    self.undo_change()
                elif choice == "6":
                    self.redo_change()
                elif choice == "7":
                    self.show_explanation_history()
                elif choice == "8":
                    self.view_manager.show_info("Thank you for using Explainable AI Visualizer!")
                    break
                else:
                    self.view_manager.show_error("Invalid choice. Please select 1-8.")
                    
            except KeyboardInterrupt:
                self.view_manager.show_info("\n\nApplication interrupted by user.")
                break
            except Exception as e:
                self.view_manager.show_error(f"Unexpected error: {str(e)}")
    
    def load_model(self):
        """Load ML model (Option 1)"""
        self.view_manager.show_section_header("LOAD ML MODEL")
        
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        if os.path.exists(models_dir):
            self.view_manager.show_info("Sample models available:")
            for file in os.listdir(models_dir):
                if file.endswith(('.pkl', '.pickle')):
                    print(f"  • {file}")
        
        path = self.view_manager.get_input("\nEnter model file path (or press Enter to cancel): ")
        
        path = path.strip().strip('"').strip("'")
        
        if not path:
            self.view_manager.show_warning("Model loading cancelled.")
            return
        
        try:
            feature_names = self.model_controller.load_model(path)
            
            self.model_subject.notify_model_loaded(feature_names)
            
            self.view_manager.show_success("Model loaded successfully!")
            self.view_manager.show_info(f"Features ({len(feature_names)}): {', '.join(feature_names)}")
            
            self.current_instance = [0.0] * len(feature_names)
            
        except Exception as e:
            self.view_manager.show_error(f"Failed to load model: {str(e)}")
    
    def view_model_metadata(self):
        """View model metadata (Option 2)"""
        self.view_manager.show_section_header("MODEL METADATA")
        
        if not self.model_controller.is_model_loaded():
            self.view_manager.show_warning("No model loaded. Please load a model first.")
            return
        
        try:
            adapter = self.model_controller.get_adapter()
            feature_names = adapter.get_feature_names()
            model_type = adapter.get_model_type()
            
            print(f"  Model Type: {model_type}")
            print(f"  Number of Features: {len(feature_names)}")
            print(f"\n  Features:")
            for i, name in enumerate(feature_names, 1):
                print(f"    {i}. {name}")
            
            print("\n" + "="*60)
            
        except Exception as e:
            self.view_manager.show_error(f"Error retrieving metadata: {str(e)}")
    
    def enter_instance(self):
        """Enter instance for prediction (Option 3)"""
        self.view_manager.show_section_header("ENTER INSTANCE FOR PREDICTION")
        
        if not self.model_controller.is_model_loaded():
            self.view_manager.show_warning("No model loaded. Please load a model first.")
            return
        
        try:
            feature_names = self.model_controller.get_adapter().get_feature_names()
            new_instance = []
            
            print("  Enter values for each feature:\n")
            
            for i, feature_name in enumerate(feature_names):
                while True:
                    try:
                        current_val = self.current_instance[i] if i < len(self.current_instance) else 0.0
                        value_str = self.view_manager.get_input(
                            f"  {feature_name} [current: {current_val:.2f}]: "
                        )
                        
                        if value_str.strip() == "":
                            value = current_val
                        else:
                            value = float(value_str)
                        
                        new_instance.append(value)
                        break
                    except ValueError:
                        self.view_manager.show_error("Invalid number. Please try again.")
            
            old_instance = self.current_instance.copy()
            self.current_instance = new_instance
            
            prediction, probabilities = self.model_controller.predict(np.array(new_instance).reshape(1, -1))
            
            self.model_subject.notify_prediction(prediction, probabilities, new_instance)
            
            print("\n" + "-"*60)
            self.view_manager.show_success("Prediction Complete!")
            print(f"  Predicted Class: {prediction}")
            if probabilities is not None:
                print(f"  Probabilities: {probabilities}")
            print("-"*60 + "\n")
            
        except Exception as e:
            self.view_manager.show_error(f"Error during prediction: {str(e)}")
    
    def generate_explanation(self):
        """Generate explanation (Option 4)"""
        self.view_manager.show_section_header("GENERATE EXPLANATION")
        
        if not self.model_controller.is_model_loaded():
            self.view_manager.show_warning("No model loaded. Please load a model first.")
            return
        
        if not self.current_instance:
            self.view_manager.show_warning("No instance entered. Please enter an instance first (Option 3).")
            return
        
        print("\n  Select Explanation Method:")
        print("    1. LIME (Local Interpretable Model-agnostic Explanations)")
        print("    2. SHAP (SHapley Additive exPlanations)")
        print("    3. Decision Path (Tree-based models only)")
        print("    4. Auto-detect best method")
        
        choice = self.view_manager.get_input("\n  Choice (1-4): ")
        
        explainer_map = {
            "1": "lime",
            "2": "shap",
            "3": "decision_path",
            "4": "auto"
        }
        
        explainer_type = explainer_map.get(choice, "auto")
        
        try:
            model = self.model_controller.get_model()
            feature_names = self.model_controller.get_adapter().get_feature_names()
            
            explainer_type_arg = None if explainer_type == "auto" else explainer_type
            
            self.current_explainer = ExplainerFactory.create(
                model=model,
                background=None,
                feature_names=feature_names,
                explainer_type=explainer_type_arg
            )
            
            if self.current_explainer is None:
                self.view_manager.show_error("Failed to create explainer. Check if required libraries are installed.")
                return
            
            self.view_manager.show_info("Generating explanation... Please wait.")
            
            instance_array = np.array(self.current_instance).reshape(1, -1)
            explanation = self.current_explainer.explain(instance_array)
            
            self.explanation_history.append({
                'type': explainer_type,
                'instance': self.current_instance.copy(),
                'explanation': explanation
            })
            
            self.view_manager.display_explanation(explanation, explainer_type, feature_names)
            
            self.view_manager.show_success("Explanation generated successfully!")
            
        except Exception as e:
            self.view_manager.show_error(f"Error generating explanation: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def undo_change(self):
        """Undo last change (Option 5)"""
        self.view_manager.show_section_header("UNDO LAST CHANGE")
        
        success = self.command_manager.undo()
        
        if success:
            self.view_manager.show_success("Last change undone successfully!")
            self.model_subject.notify({'type': 'instance_changed'})
        else:
            self.view_manager.show_warning("Nothing to undo.")
    
    def redo_change(self):
        """Redo last change (Option 6)"""
        self.view_manager.show_section_header("REDO LAST CHANGE")
        
        success = self.command_manager.redo()
        
        if success:
            self.view_manager.show_success("Last change redone successfully!")
            self.model_subject.notify({'type': 'instance_changed'})
        else:
            self.view_manager.show_warning("Nothing to redo.")
    
    def show_explanation_history(self):
        """Show explanation history (Option 7)"""
        self.view_manager.show_section_header("EXPLANATION HISTORY")
        
        if not self.explanation_history:
            self.view_manager.show_warning("No explanations generated yet.")
            return
        
        print(f"  Total Explanations: {len(self.explanation_history)}\n")
        
        for i, entry in enumerate(self.explanation_history, 1):
            print(f"  {i}. Method: {entry['type'].upper()}")
            print(f"     Instance: {entry['instance'][:3]}... (showing first 3 features)")
            print()


def main():
    """Main entry point for CLI application"""
    app = ExplainableAICLI()
    app.run()


if __name__ == "__main__":
    main()
