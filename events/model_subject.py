# events/model_subject.py
from typing import List, Dict, Any
from .interfaces import Observer

class ModelSubject:
    """
    Purpose: Notify UI observers when predictions or explanations change (Observable).
    """
    _observers: List[Observer] = []

    def attach(self, observer: Observer):
        """Maintain observer list: add an observer."""
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer: Observer):
        """Maintain observer list: remove an observer."""
        try:
            self._observers.remove(observer)
        except ValueError:
            pass # Observer was not attached

    def notify(self, data: Dict[str, Any]):
        """Notify on change: send data to all attached observers."""
        for observer in self._observers:
            observer.update(data)