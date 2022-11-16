from models.byOnlyMeans import ByOnlyMeans
from models.useDecisionTree import UseDecisionTree
from processor import Processor
from utils import get_only_not_black, get_all

fourChildrenMeansRoot = {
    0: {
        'modelClass': ByOnlyMeans,
        'processor': Processor(when_grey_std=20),
        'goto': {
            'GreyGreen': 1,
            'BlueCyanViolet': 2,
            'BrownOrangeRedYellow': 3,
            'GreenYellow': 4
        },
        'union': {
            'GreyGreen': ['Black', 'Grey', 'White', 'Green'],
            'BlueCyanViolet': ['Blue', 'Cyan', 'Violet'],
            'BrownOrangeRedYellow': ['Brown', 'Orange', 'Red', 'Yellow'],
            'GreenYellow': ['Green', 'Yellow'],
        },
        'classes': ['GreyGreen', 'BlueCyanViolet', 'BrownOrangeRedYellow', 'GreenYellow'],
        'kwargs': {}
    },
    1: {
        'modelClass': ByOnlyMeans,
        'processor': Processor(when_grey_std=7),
        'goto': {},
        'union': {},
        'classes': ['Black', 'Grey', 'White', 'Green'],
        'kwargs': {}
    },
    2: {
        'modelClass': ByOnlyMeans,
        'processor': Processor(when_grey_std=40),
        'goto': {},
        'union': {},
        'classes': ['Blue', 'Cyan', 'Violet'],
        'kwargs': {}
    },
    3: {
        'modelClass': ByOnlyMeans,
        'processor': Processor(when_grey_std=30),
        'goto': {},
        'union': {},
        'classes': ['Brown', 'Orange', 'Red', 'Yellow'],
        'kwargs': {}
    },
    4: {
        'modelClass': ByOnlyMeans,
        'processor': Processor(when_grey_std=20),
        'goto': {},
        'union': {},
        'classes': ['Green', 'Yellow'],
        'kwargs': {}
    }
}

greyAndColour = {
    0: {
        
    }
}