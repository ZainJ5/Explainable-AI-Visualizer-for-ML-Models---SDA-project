"""
Model Loader Strategy Pattern
Defines different strategies for loading ML models with various compatibility modes
"""

from abc import ABC, abstractmethod
from typing import Any
import pickle
import io

try:
    import pickle5
    HAS_PICKLE5 = True
except ImportError:
    HAS_PICKLE5 = False

try:
    import dill
    HAS_DILL = True
except ImportError:
    HAS_DILL = False

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


class ModelLoaderStrategy(ABC):
    """
    Strategy Interface: Defines contract for different model loading strategies
    Each concrete strategy implements a specific loading mechanism
    """
    
    @abstractmethod
    def load(self, file_path: str, file_content: bytes) -> Any:
        """
        Load a model using this strategy
        
        Args:
            file_path: Path to the model file
            file_content: Raw bytes content of the file
            
        Returns:
            Loaded model object
            
        Raises:
            Exception: If loading fails with this strategy
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this strategy"""
        pass


class StandardPickleStrategy(ModelLoaderStrategy):
    """Load using standard pickle with specific encoding"""
    
    def __init__(self, encoding: str = 'latin1'):
        self.encoding = encoding
    
    def load(self, file_path: str, file_content: bytes) -> Any:
        return pickle.loads(file_content, encoding=self.encoding)
    
    def get_name(self) -> str:
        return f"Standard pickle ({self.encoding})"


class Pickle5Strategy(ModelLoaderStrategy):
    """Load using pickle5 for Python 3.6-3.7 compatibility"""
    
    def __init__(self, encoding: str = 'latin1'):
        self.encoding = encoding
    
    def load(self, file_path: str, file_content: bytes) -> Any:
        if not HAS_PICKLE5:
            raise ImportError("pickle5 not installed")
        return pickle5.loads(file_content, encoding=self.encoding)
    
    def get_name(self) -> str:
        return f"pickle5 ({self.encoding})"


class DillStrategy(ModelLoaderStrategy):
    """Load using dill for complex objects"""
    
    def load(self, file_path: str, file_content: bytes) -> Any:
        if not HAS_DILL:
            raise ImportError("dill not installed")
        return dill.loads(file_content)
    
    def get_name(self) -> str:
        return "dill"


class JoblibMemoryStrategy(ModelLoaderStrategy):
    """Load using joblib from memory (BEST for sklearn models)"""
    
    def load(self, file_path: str, file_content: bytes) -> Any:
        if not HAS_JOBLIB:
            raise ImportError("joblib not installed")
        return joblib.load(io.BytesIO(file_content))
    
    def get_name(self) -> str:
        return "joblib (memory)"


class JoblibFileStrategy(ModelLoaderStrategy):
    """Load using joblib directly from file"""
    
    def load(self, file_path: str, file_content: bytes) -> Any:
        if not HAS_JOBLIB:
            raise ImportError("joblib not installed")
        return joblib.load(file_path)
    
    def get_name(self) -> str:
        return "joblib (file)"


class CustomUnpicklerStrategy(ModelLoaderStrategy):
    """Load using custom unpickler with module resolution"""
    
    def load(self, file_path: str, file_content: bytes) -> Any:
        class CompatibilityUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                """Override to handle module name changes"""
                if module.startswith('sklearn.'):
                    try:
                        return super().find_class(module, name)
                    except (ModuleNotFoundError, AttributeError):
                        pass
                
                try:
                    return super().find_class(module, name)
                except Exception:
                    try:
                        mod = __import__(module, fromlist=[name])
                        return getattr(mod, name)
                    except Exception:
                        raise
        
        for encoding in ['latin1', 'bytes', 'ASCII', None]:
            try:
                unpickler = CompatibilityUnpickler(io.BytesIO(file_content))
                if encoding:
                    unpickler.encoding = encoding
                return unpickler.load()
            except Exception:
                continue
        
        raise ValueError("Custom unpickler failed with all encodings")
    
    def get_name(self) -> str:
        return "Custom unpickler"


class ModelLoaderContext:
    """
    Context class that uses a ModelLoaderStrategy
    Tries multiple strategies in order until one succeeds
    """
    
    def __init__(self):
        self.strategies = self._initialize_strategies()
    
    def _initialize_strategies(self) -> list:
        """Initialize all available loading strategies in priority order"""
        strategies = []
        
        if HAS_JOBLIB:
            strategies.append(JoblibMemoryStrategy())
            strategies.append(JoblibFileStrategy())
        
        strategies.extend([
            StandardPickleStrategy('latin1'),
            StandardPickleStrategy('bytes'),
            StandardPickleStrategy('ASCII'),
        ])
        
        if HAS_PICKLE5:
            strategies.append(Pickle5Strategy('latin1'))
            strategies.append(Pickle5Strategy())
        
        if HAS_DILL:
            strategies.append(DillStrategy())
        
        strategies.append(CustomUnpicklerStrategy())
        
        return strategies
    
    def load_model(self, file_path: str) -> Any:
        """
        Load model by trying all strategies until one succeeds
        
        Args:
            file_path: Path to model file
            
        Returns:
            Loaded model object
            
        Raises:
            ValueError: If all strategies fail
        """
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        errors = []
        
        for strategy in self.strategies:
            try:
                model = strategy.load(file_path, file_content)
                print(f"✓ Successfully loaded model using: {strategy.get_name()}")
                return model
            except Exception as e:
                errors.append(f"{strategy.get_name()}: {str(e)[:50]}")
        
        error_msg = "Failed to load model with all strategies:\n" + "\n".join(errors)
        error_msg += "\n\nSuggestion: Try regenerating the model with: python models/create_sample_models.py"
        raise ValueError(error_msg)
