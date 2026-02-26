import pandas as pd
import os

class PipelineNode:
    """Base class for all pipeline nodes."""
    __array_priority__ = 10000
    __pandas_priority__ = 10000

    def process(self, data):
        raise NotImplementedError("Subclasses must implement process()")

    def __or__(self, other):
        """Allows chaining nodes: Node1() | Node2()"""
        if isinstance(other, PipelineNode):
            return PipelineSequence([self, other])
        elif isinstance(other, PipelineSequence):
            return PipelineSequence([self] + other.nodes)
        return NotImplemented

    def __ror__(self, other):
        """Allows piping data: data | Node()"""
        # If the left operand is not a PipelineNode, process it
        return self.process(other)


class PipelineSequence(PipelineNode):
    """Represents a chain of pipeline nodes."""
    def __init__(self, nodes):
        self.nodes = nodes

    def process(self, data):
        result = data
        for node in self.nodes:
            result = node.process(result)
        return result

    def __or__(self, other):
        if isinstance(other, PipelineNode):
            if isinstance(other, PipelineSequence):
                return PipelineSequence(self.nodes + other.nodes)
            return PipelineSequence(self.nodes + [other])
        return NotImplemented

    def __ror__(self, other):
        # Data | PipelineSequence
        return self.process(other)


class Load(PipelineNode):
    """Loads a pandas DataFrame from a CSV file path or passes through an existing DataFrame."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def process(self, data):
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, str):
            if not os.path.exists(data):
                raise FileNotFoundError(f"File not found: {data}")
            return pd.read_csv(data, **self.kwargs)
        else:
            raise ValueError("Load node expects a file path (str) or a pandas DataFrame.")


class Train(PipelineNode):
    """Trains an AutoModel over the incoming DataFrame."""
    def __init__(self, target, task="auto", mode="fast", **kwargs):
        self.target = target
        self.task = task
        self.mode = mode
        self.kwargs = kwargs

    def process(self, data):
        from .automodel import AutoModel
        
        # Handle the case where someone pipes a trained model, then a Train node...
        # that doesn't make sense. Train node expects data. 
        # But if the user inputs a string path without Load(), we can support that using helpers.train_model logic,
        # but let's encourage Load() | Train() or DataFrame | Train().
        
        # If Data is str or dict, AutoModel can handle it in train(), so we just pass it along.
        model = AutoModel(
            task=self.task,
            mode=self.mode,
            **self.kwargs
        )
        model.train(data, target=self.target)
        return model


class Predict(PipelineNode):
    """Uses a trained model to make predictions on the incoming data."""
    def __init__(self, model):
        self.model = model

    def process(self, data):
        return self.model.predict(data)


class Evaluate(PipelineNode):
    """Evaluates the incoming model's internal score on the metric."""
    def __init__(self, model=None):
        self.model = model

    def process(self, incoming):
        from .automodel import AutoModel
        
        # If piped model | Evaluate()
        if isinstance(incoming, AutoModel):
            return incoming.score()
        # If Evaluate(model) was created and data piped in: data | Evaluate(model)
        # However, score() doesn't take new data yet.
        # We can just return the score of the model.
        if self.model is not None and isinstance(self.model, AutoModel):
            return self.model.score()
        
        raise ValueError("Evaluate node expects a trained AutoModel.")
