# Smart Trash Truck Routing System

A Python-based intelligent routing system for optimizing garbage collection routes using advanced algorithms and real-time bin status monitoring.

## Features

- **Real-time Bin Monitoring**: Track fill levels and prioritize collections
- **Dynamic Route Optimization**: Calculate efficient routes based on bin status
- **Network Analysis**: View and analyze the complete network, clusters, and MST
- **Interactive Visualization**: Nord-themed visualizations of routes and network analysis
- **Multi-truck Support**: Handle multiple trucks with different capacities
- **Priority-based Assignment**: Automatically assign bins to trucks based on urgency

## Requirements

```
pandas>=2.2.3
numpy>=2.2.5
networkx>=3.4.2
matplotlib>=3.10.3
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ruinedcodes/Smart-Trash-Truck-Routing.git
cd Smart-Trash-Truck-Routing
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main program:
```bash
python smart_trash_routing.py
```

The system provides three test datasets:
1. Basic Test (6 bins, 1 truck)
2. Constraint Test (12 bins, 2 trucks)
3. Optimization Test (24 bins, 3 trucks)

### Main Features:

1. **System Status**
   - View bin fill levels and priorities
   - Monitor truck capacities and current loads
   - Track high-priority bins

2. **Route Management**
   - Calculate optimal routes
   - View current route assignments
   - Analyze route statistics

3. **Network Analysis**
   - Visualize complete network
   - View bin clusters
   - Analyze Minimum Spanning Tree

4. **Real-time Operations**
   - Update bin fill levels
   - Simulate real-time updates
   - View recent changes

## Project Structure

```
SmartTrashTruckRoutingSys/
├── data/                      # Test datasets
│   ├── basic_test/           # 6 bins, 1 truck
│   ├── constraint_test/      # 12 bins, 2 trucks
│   └── optimization_test/    # 24 bins, 3 trucks
├── sttrs_package/            # Core package
│   ├── core/                 # Core functionality
├── setup.py                  # Setup for Main application
├── requirements.txt          # Project dependencies
└── smart_trash_routing.py    # Main application
```

## Algorithms

The system uses several algorithms for optimization:
- **Bin Assignment**: Priority-based bin-to-truck assignment
- **Route Optimization**: Modified TSP solver with clustering
- **Network Analysis**: MST and clustering algorithms
- **Dynamic Updates**: Real-time route recalculation

## UI Features

- Clean, modern terminal interface
- Nord color theme throughout
- Intuitive navigation
- Consistent left-aligned menus
- Interactive visualizations

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
