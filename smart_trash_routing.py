import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from queue import PriorityQueue
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import heapq
from datetime import datetime
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import colors from the package
from sttrs_package.core.colors import *

@dataclass
class Bin:
    id: str
    latitude: float
    longitude: float
    fill_level: float
    capacity: float
    last_updated: datetime = None
    
    @property
    def fill_percentage(self) -> float:
        return (self.fill_level / self.capacity) * 100
    
    @property
    def is_high_priority(self) -> bool:
        return self.fill_percentage >= 80

    def update_fill_level(self, new_level: float, timestamp: datetime):
        self.fill_level = new_level
        self.last_updated = timestamp

@dataclass
class Truck:
    id: str
    capacity: float
    current_load: float
    current_location: str
    route: List[str] = None
    
    @property
    def available_capacity(self) -> float:
        return self.capacity - self.current_load

class BPlusNode:
    def __init__(self, leaf=True):
        self.leaf = leaf
        self.keys = []
        self.children = []
        self.bins = {}  # Only used in leaf nodes

class BPlusTree:
    def __init__(self, order=4):
        self.root = BPlusNode(leaf=True)
        self.order = order

    def insert(self, key: str, bin_obj: Bin):
        if len(self.root.keys) == (2 * self.order) - 1:
            new_root = BPlusNode(leaf=False)
            new_root.children.append(self.root)
            self._split_child(new_root, 0)
            self.root = new_root
        self._insert_non_full(self.root, key, bin_obj)

    def _split_child(self, parent: BPlusNode, index: int):
        order = self.order
        child = parent.children[index]
        new_node = BPlusNode(leaf=child.leaf)
        
        parent.keys.insert(index, child.keys[order - 1])
        parent.children.insert(index + 1, new_node)
        
        new_node.keys = child.keys[order:]
        child.keys = child.keys[:order - 1]
        
        if child.leaf:
            new_node.bins = {k: child.bins[k] for k in new_node.keys}
            child.bins = {k: child.bins[k] for k in child.keys}

    def _insert_non_full(self, node: BPlusNode, key: str, bin_obj: Bin):
        i = len(node.keys) - 1
        if node.leaf:
            while i >= 0 and key < node.keys[i]:
                i -= 1
            node.keys.insert(i + 1, key)
            node.bins[key] = bin_obj
        else:
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            if len(node.children[i].keys) == (2 * self.order) - 1:
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            self._insert_non_full(node.children[i], key, bin_obj)

    def get(self, key: str) -> Optional[Bin]:
        node = self.root
        while not node.leaf:
            i = 0
            while i < len(node.keys) and key >= node.keys[i]:
                i += 1
            node = node.children[i]
        
        if key in node.bins:
            return node.bins[key]
        return None

class SmartTrashRouter:
    def __init__(self, bins_file: str, roads_file: str, trucks_file: str):
        self.bins: Dict[str, Bin] = {}
        self.bin_tree = BPlusTree(order=4)
        self.trucks: List[Truck] = []
        self.graph = nx.Graph()
        self.mst = None  # Minimum Spanning Tree
        self.clusters = []  # Store identified clusters
        self.load_data(bins_file, roads_file, trucks_file)
        self.build_mst()
        self.identify_clusters()

    def load_data(self, bins_file: str, roads_file: str, trucks_file: str):
        # Load bins data
        bins_df = pd.read_csv(bins_file)
        for _, row in bins_df.iterrows():
            bin_obj = Bin(
                id=str(row['bin_id']).strip(),  # Strip whitespace from bin ID
                latitude=float(row['latitude']),
                longitude=float(row['longitude']),
                fill_level=float(row['fill_level']),
                capacity=float(row['capacity']),
                last_updated=datetime.now()
            )
            self.bins[bin_obj.id] = bin_obj
            self.bin_tree.insert(bin_obj.id, bin_obj)
            self.graph.add_node(bin_obj.id, bin=bin_obj)
        
        # Load roads data
        roads_df = pd.read_csv(roads_file)
        for _, row in roads_df.iterrows():
            self.graph.add_edge(
                str(row['from_bin']).strip(),  # Strip whitespace from bin IDs
                str(row['to_bin']).strip(),
                weight=float(row['distance'])
            )
        
        # Load trucks data
        trucks_df = pd.read_csv(trucks_file)
        for _, row in trucks_df.iterrows():
            self.trucks.append(
                Truck(
                    id=str(row['truck_id']).strip(),  # Strip whitespace from truck ID
                    capacity=float(row['capacity']),
                    current_load=float(row['current_load']),
                    current_location=str(row['start_location']).strip()  # Strip whitespace from location
                )
            )

    def build_mst(self):
        """Build Minimum Spanning Tree using Kruskal's Algorithm"""
        self.mst = nx.minimum_spanning_tree(self.graph, algorithm='kruskal')

    def identify_clusters(self, distance_threshold: float = 1.0):
        """Identify clusters of bins using connected components and proximity"""
        # First, get connected components from the road network
        components = list(nx.connected_components(self.graph))
        
        # For each component, identify sub-clusters based on proximity
        self.clusters = []
        for component in components:
            component = list(component)
            if len(component) <= 1:
                continue
            
            # Get coordinates for bins in this component
            coords = np.array([[self.bins[b].latitude, self.bins[b].longitude] 
                              for b in component])
            
            # Calculate distance matrix for this component
            n = len(component)
            distance_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    # Use road network distance if available, otherwise Euclidean
                    try:
                        dist = nx.shortest_path_length(
                            self.graph,
                            component[i],
                            component[j],
                            weight='weight'
                        )
                    except (nx.NetworkXNoPath, nx.NetworkXError):
                        dist = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
                    distance_matrix[i, j] = distance_matrix[j, i] = dist
            
            # Group bins within distance threshold
            visited = set()
            for i in range(n):
                if i in visited:
                    continue
                
                cluster = {i}
                visited.add(i)
                
                # Find all bins within threshold distance
                for j in range(n):
                    if j not in visited and distance_matrix[i, j] <= distance_threshold:
                        cluster.add(j)
                        visited.add(j)
                
                if len(cluster) > 1:  # Only keep clusters with multiple bins
                    self.clusters.append([component[i] for i in cluster])
        
        # If no clusters were found, create clusters based on proximity only
        if not self.clusters:
            bin_ids = list(self.bins.keys())
            n = len(bin_ids)
            if n > 1:
                coords = np.array([[self.bins[b].latitude, self.bins[b].longitude] 
                                  for b in bin_ids])
                
                # Create clusters of nearby bins
                visited = set()
                for i in range(n):
                    if i in visited:
                        continue
                    
                    cluster = {i}
                    visited.add(i)
                    
                    # Find nearby bins
                    for j in range(n):
                        if j not in visited:
                            dist = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
                            if dist <= distance_threshold:
                                cluster.add(j)
                                visited.add(j)
                    
                    if len(cluster) > 1:
                        self.clusters.append([bin_ids[i] for i in cluster])
        
        # If still no clusters, create one cluster per connected component
        if not self.clusters:
            self.clusters = [list(comp) for comp in components if len(comp) > 1]
        
        # If still no clusters, put all bins in one cluster
        if not self.clusters and len(self.bins) > 1:
            self.clusters = [list(self.bins.keys())]

    def optimize_cluster_sequence(self, cluster: List[str], start_bin: str) -> List[str]:
        """Use Dynamic Programming to optimize the sequence of bins within a cluster"""
        n = len(cluster)
        if n <= 2:
            return cluster
        
        # Create distance matrix for the cluster
        dist_matrix = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    try:
                        dist = nx.shortest_path_length(
                            self.graph,
                            cluster[i],
                            cluster[j],
                            weight='weight'
                        )
                    except nx.NetworkXNoPath:
                        dist = float('inf')
                    dist_matrix[(cluster[i], cluster[j])] = dist
        
        # Dynamic Programming solution
        # State: (current_bin, visited_bins)
        dp = {}
        def solve(current: str, visited: frozenset) -> Tuple[float, List[str]]:
            if len(visited) == n:
                return 0, []
            
            state = (current, visited)
            if state in dp:
                return dp[state]
            
            min_cost = float('inf')
            best_path = []
            
            for next_bin in cluster:
                if next_bin not in visited:
                    cost = dist_matrix.get((current, next_bin), float('inf'))
                    remaining_cost, remaining_path = solve(
                        next_bin,
                        frozenset(visited | {next_bin})
                    )
                    total_cost = cost + remaining_cost
                    
                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_path = [next_bin] + remaining_path
            
            dp[state] = (min_cost, best_path)
            return min_cost, best_path
        
        _, optimal_path = solve(start_bin, frozenset({start_bin}))
        return [start_bin] + optimal_path

    def update_bin_status(self, bin_id: str, new_fill_level: float):
        """Update bin status and potentially trigger route recalculation"""
        bin_id = bin_id.strip()  # Strip whitespace from bin ID
        bin_obj = self.bin_tree.get(bin_id)
        if bin_obj:
            old_priority = bin_obj.is_high_priority
            bin_obj.update_fill_level(new_fill_level, datetime.now())
            
            # If priority changed, trigger route recalculation
            if old_priority != bin_obj.is_high_priority:
                self.recalculate_routes()

    def recalculate_routes(self):
        """Recalculate routes based on current bin statuses"""
        assignments = self.assign_bins_to_trucks()
        for truck_id, assigned_bins in assignments.items():
            truck = next(t for t in self.trucks if t.id == truck_id)
            route = self.find_optimal_route(truck, assigned_bins)
            truck.route = route

    def get_priority_bins(self) -> List[Bin]:
        """Return bins sorted by priority (fill level percentage)"""
        return sorted(
            self.bins.values(),
            key=lambda x: x.fill_percentage,
            reverse=True
        )

    def find_optimal_route(self, truck: Truck, target_bins: List[str]) -> List[str]:
        """Find the shortest path that covers all target bins"""
        if not target_bins:
            return []
        
        # Clean input data and precompute distances
        current = truck.current_location.strip()
        remaining_bins = {bin_id.strip() for bin_id in target_bins}
        route = [current]
        
        # Precompute shortest paths between all nodes
        try:
            distances = dict(nx.all_pairs_shortest_path_length(self.graph, weight='weight'))
        except Exception:
            distances = {}  # Fallback if path computation fails
        
        # Process bins cluster by cluster
        while remaining_bins:
            # Find the cluster containing the most remaining bins
            current_cluster = None
            max_overlap = 0
            
            for cluster in self.clusters:
                clean_cluster = {bin_id.strip() for bin_id in cluster}
                overlap = len(remaining_bins.intersection(clean_cluster))
                if overlap > max_overlap:
                    max_overlap = overlap
                    current_cluster = clean_cluster
            
            if current_cluster and max_overlap > 0:
                # Handle bins in current cluster
                cluster_bins = list(remaining_bins.intersection(current_cluster))
                
                # Simple nearest neighbor for cluster bins
                while cluster_bins:
                    # Find nearest bin in cluster
                    nearest_bin = None
                    min_distance = float('inf')
                    
                    for bin_id in cluster_bins:
                        if current in distances and bin_id in distances[current]:
                            distance = distances[current][bin_id]
                            if distance < min_distance:
                                min_distance = distance
                                nearest_bin = bin_id
                    
                    if nearest_bin:
                        if nearest_bin != route[-1]:  # Avoid duplicates
                            route.append(nearest_bin)
                        current = nearest_bin
                        cluster_bins.remove(nearest_bin)
                        remaining_bins.remove(nearest_bin)
                    else:
                        # If no path found, just add the first bin
                        next_bin = cluster_bins[0]
                        if next_bin != route[-1]:  # Avoid duplicates
                            route.append(next_bin)
                        current = next_bin
                        cluster_bins.remove(next_bin)
                        remaining_bins.remove(next_bin)
            else:
                # Handle remaining bins using nearest neighbor
                nearest_bin = None
                min_distance = float('inf')
                
                for bin_id in remaining_bins:
                    if current in distances and bin_id in distances[current]:
                        distance = distances[current][bin_id]
                        if distance < min_distance:
                            min_distance = distance
                            nearest_bin = bin_id
                
                if nearest_bin:
                    if nearest_bin != route[-1]:  # Avoid duplicates
                        route.append(nearest_bin)
                    current = nearest_bin
                    remaining_bins.remove(nearest_bin)
                else:
                    # If no path found, just add the first remaining bin
                    next_bin = next(iter(remaining_bins))
                    if next_bin != route[-1]:  # Avoid duplicates
                        route.append(next_bin)
                    current = next_bin
                    remaining_bins.remove(next_bin)
        
        return route

    def assign_bins_to_trucks(self) -> Dict[str, List[str]]:
        """Assign bins to trucks based on capacity and priority"""
        try:
            assignments = {truck.id: [] for truck in self.trucks}
            if not self.bins or not self.trucks:
                return assignments

            # Reset truck loads before assignment
            for truck in self.trucks:
                truck.current_load = 0.0

            # Get bins sorted by priority
            priority_bins = self.get_priority_bins()
            if not priority_bins:
                return assignments

            # First assign high priority bins
            high_priority = [bin_obj for bin_obj in priority_bins if bin_obj.is_high_priority]
            remaining = [bin_obj for bin_obj in priority_bins if not bin_obj.is_high_priority]
            
            # Sort trucks by available capacity
            available_trucks = sorted(
                self.trucks,
                key=lambda x: x.available_capacity,
                reverse=True
            )
            
            # Assign high priority bins first
            for bin_obj in high_priority:
                # Find the nearest truck with sufficient capacity
                assigned = False
                for truck in available_trucks:
                    if truck.available_capacity >= bin_obj.fill_level:
                        assignments[truck.id].append(bin_obj.id)
                        truck.current_load += bin_obj.fill_level
                        assigned = True
                        break
                
                if not assigned:
                    # If no truck has enough capacity, assign to the truck with most capacity
                    truck = max(available_trucks, key=lambda x: x.available_capacity)
                    assignments[truck.id].append(bin_obj.id)
                    truck.current_load += bin_obj.fill_level
            
            # Assign remaining bins using a simple round-robin approach
            if remaining:
                truck_index = 0
                for bin_obj in remaining:
                    # Try to find a truck with enough capacity
                    original_index = truck_index
                    while True:
                        if available_trucks[truck_index].available_capacity >= bin_obj.fill_level:
                            break
                        truck_index = (truck_index + 1) % len(available_trucks)
                        if truck_index == original_index:
                            # If we've checked all trucks, use the one with most capacity
                            truck_index = max(range(len(available_trucks)), 
                                            key=lambda i: available_trucks[i].available_capacity)
                            break
                    
                    # Assign the bin
                    assignments[available_trucks[truck_index].id].append(bin_obj.id)
                    available_trucks[truck_index].current_load += bin_obj.fill_level
                    truck_index = (truck_index + 1) % len(available_trucks)
            
            return assignments
        except Exception as e:
            print(center_text(f"Error in bin assignment: {str(e)}"))
            return {truck.id: [] for truck in self.trucks}

    def visualize_route(self, route: List[str], title: str = "Route Visualization"):
        """Visualize the route on a map"""
        plt.figure(figsize=(12, 8))
        
        # Clean bin IDs and create position mapping
        pos = {bin_id.strip(): (self.bins[bin_id.strip()].longitude, self.bins[bin_id.strip()].latitude) 
               for bin_id in self.bins}
        
        # Draw all edges (roads) in light frost color
        nx.draw_networkx_edges(self.graph, pos, alpha=0.2, edge_color='#81a1c1')  # STEEL color
        
        # Draw MST edges in sage color
        nx.draw_networkx_edges(self.mst, pos, edge_color='#8fbcbb', alpha=0.5)  # SAGE color
        
        # Draw clusters with frost colors
        cluster_colors = ['#8fbcbb', '#88c0d0', '#81a1c1']  # SAGE, FROST, STEEL colors
        for cluster, color in zip(self.clusters, cluster_colors * (len(self.clusters) // 3 + 1)):
            # Clean cluster bin IDs
            clean_cluster = [bin_id.strip() for bin_id in cluster]
            nx.draw_networkx_nodes(
                self.graph,
                pos,
                nodelist=clean_cluster,
                node_color=color,
                alpha=0.3,
                node_size=500
            )
        
        if route:
            # Clean route bin IDs and create edges
            clean_route = [bin_id.strip() for bin_id in route]
            route_edges = list(zip(clean_route[:-1], clean_route[1:]))
            
            # Draw route edges with arrows in purple
            nx.draw_networkx_edges(
                self.graph,
                pos,
                edgelist=route_edges,
                edge_color='#b48ead',  # PURPLE color
                width=2,
                arrows=True,
                arrowsize=20
            )
            
            # Add edge labels for route distances
            edge_labels = {}
            for i in range(len(clean_route)-1):
                try:
                    dist = nx.shortest_path_length(
                        self.graph,
                        clean_route[i],
                        clean_route[i+1],
                        weight='weight'
                    )
                    edge_labels[(clean_route[i], clean_route[i+1])] = f'{dist:.1f}'
                except nx.NetworkXNoPath:
                    continue
            
            nx.draw_networkx_edge_labels(
                self.graph,
                pos,
                edge_labels=edge_labels,
                font_size=8,
                font_color='#4c566a'  # Dark gray from Nord theme
            )
        
        # Draw nodes with priority coloring
        node_colors = ['#bf616a' if self.bins[n.strip()].is_high_priority else '#d8dee9'  # RED for high priority, light gray for normal
                      for n in self.graph.nodes()]
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_size=700,
            node_color=node_colors
        )
        
        # Add node labels with fill levels
        node_labels = {
            n: f'{n}\n{self.bins[n].fill_percentage:.0f}%'
            for n in self.graph.nodes()
        }
        nx.draw_networkx_labels(
            self.graph,
            pos,
            labels=node_labels,
            font_size=8,
            font_weight='bold',
            font_color='#2e3440'  # Nord polar night color
        )
        
        plt.title(title, color='#2e3440', pad=20)  # Nord polar night color
        plt.axis('on')
        plt.grid(True, color='#eceff4', alpha=0.2)  # Light grid lines
        
        # Add legend with Nord theme colors
        legend_elements = [
            plt.Line2D([0], [0], color='#81a1c1', alpha=0.2, label='Roads'),
            plt.Line2D([0], [0], color='#8fbcbb', alpha=0.5, label='MST'),
            plt.Line2D([0], [0], color='#b48ead', label='Route'),
            plt.scatter([0], [0], c='#bf616a', alpha=1, label='High Priority'),
            plt.scatter([0], [0], c='#d8dee9', alpha=1, label='Normal Priority')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Set background color
        plt.gca().set_facecolor('#ffffff')  # White background
        plt.gcf().set_facecolor('#ffffff')
        
        plt.show()

    def view_mst(self):
        """Display the Minimum Spanning Tree"""
        print_centered_box("Minimum Spanning Tree Visualization")
        
        if self.mst:
            plt.figure(figsize=(12, 8))
            pos = {bin_id: (self.bins[bin_id].longitude, self.bins[bin_id].latitude) 
                   for bin_id in self.bins}
            
            # Draw MST edges with weight labels in sage color
            nx.draw_networkx_edges(self.mst, pos, edge_color='#8fbcbb', width=2)  # SAGE color
            
            # Add edge labels for MST
            edge_labels = {}
            for (u, v) in self.mst.edges():
                edge_labels[(u, v)] = f'{self.mst[u][v]["weight"]:.1f}'
            
            nx.draw_networkx_edge_labels(
                self.mst,
                pos,
                edge_labels=edge_labels,
                font_size=8,
                font_color='#4c566a'  # Dark gray from Nord theme
            )
            
            # Draw nodes in frost color
            nx.draw_networkx_nodes(
                self.mst,
                pos,
                node_size=700,
                node_color='#88c0d0'  # FROST color
            )
            
            # Add node labels with fill levels
            node_labels = {
                n: f'{n}\n{self.bins[n].fill_percentage:.0f}%'
                for n in self.mst.nodes()
            }
            nx.draw_networkx_labels(
                self.mst,
                pos,
                labels=node_labels,
                font_size=8,
                font_weight='bold',
                font_color='#2e3440'  # Nord polar night color
            )
            
            plt.title("Minimum Spanning Tree", color='#2e3440', pad=20)  # Nord polar night color
            plt.axis('on')
            plt.grid(True, color='#eceff4', alpha=0.2)  # Light grid lines
            
            # Add legend with Nord theme colors
            legend_elements = [
                plt.Line2D([0], [0], color='#8fbcbb', label='MST Edge'),
                plt.scatter([0], [0], c='#88c0d0', alpha=1, label='Bin')
            ]
            plt.legend(handles=legend_elements, loc='upper right')
            
            # Set background color
            plt.gca().set_facecolor('#ffffff')  # White background
            plt.gcf().set_facecolor('#ffffff')
            
            plt.show()
        else:
            print(center_text("MST not available. Please initialize the router first."))
        
        center_input("\nPress Enter to continue...")

def get_terminal_size():
    """Get the terminal size"""
    try:
        from shutil import get_terminal_size as _get_terminal_size
        columns, rows = _get_terminal_size()
    except:
        # Fallback for Windows
        try:
            from os import get_terminal_size as _get_terminal_size
            columns, rows = _get_terminal_size()
        except:
            columns, rows = 80, 24
    return columns, rows

def strip_ansi(text):
    """Remove ANSI escape codes from text for length calculation."""
    import re
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def center_text(text, width=None):
    """Center text in the terminal, accounting for ANSI color codes."""
    if width is None:
        width, _ = get_terminal_size()
    
    # Get the visible length of text (excluding ANSI codes)
    visible_length = len(strip_ansi(text))
    
    # Calculate padding
    padding = (width - visible_length) // 2
    if padding > 0:
        return ' ' * padding + text
    return text

def center_input(prompt):
    """Get centered input with proper cursor placement"""
    width, _ = get_terminal_size()
    padding = (width - len(prompt)) // 2
    if padding > 0:
        print(' ' * padding, end='')
    return input(apply_style(prompt, SAGE))

def print_centered_box(title, items=None, width=None):
    """Print a centered menu with left-aligned items"""
    if width is None:
        width, _ = get_terminal_size()
    
    # Print title
    print()  # Add spacing before title
    print(center_text(apply_style(title, PURPLE)))
    
    if items:
        # Find the longest item length for centering calculation
        max_length = max(len(strip_ansi(item)) for item in items)
        base_padding = (width - max_length) // 2
        
        print()  # Add spacing after title
        for item in items:
            # All items use the same padding for consistent left alignment
            print(' ' * base_padding + apply_style(item, TEXT_LIGHTER))
        print()  # Add spacing after menu

def clear_screen():
    """Clear the terminal screen"""
    from sttrs_package.core.colors import clear_screen as clear
    clear()

def print_header():
    """Print the application header"""
    clear_screen()
    width, _ = get_terminal_size()
    print_centered_box("SMART TRASH TRUCK ROUTING SYSTEM", width=width)
    print()  # Single newline after header

def print_main_menu():
    """Print the main menu options"""
    items = [
        "1. View System Status",
        "2. Route Management",
        "3. Network Analysis",
        "4. Real-time Operations",
        "5. Switch Test Data",
        "q. Quit Program"
    ]
    print_centered_box("Main Menu", items)

def print_status_menu():
    """Print the status menu options"""
    items = [
        "1. View Bins Status",
        "2. View Trucks Status",
        "3. View High Priority Bins",
        "b. Back to Main Menu",
        "q. Quit Program"
    ]
    print_centered_box("System Status", items)

def print_route_menu():
    """Print the route management menu options"""
    items = [
        "1. Calculate and View Routes",
        "2. View Current Routes",
        "3. View Route Statistics",
        "b. Back to Main Menu",
        "q. Quit Program"
    ]
    print_centered_box("Route Management", items)

def print_network_menu():
    """Print the network analysis menu options"""
    items = [
        "1. View Complete Network",
        "2. View Clusters",
        "3. View Minimum Spanning Tree",
        "b. Back to Main Menu",
        "q. Quit Program"
    ]
    print_centered_box("Network Analysis", items)

def print_operations_menu():
    """Print the real-time operations menu options"""
    items = [
        "1. Update Bin Fill Level",
        "2. Simulate Updates",
        "3. View Recent Updates",
        "b. Back to Main Menu",
        "q. Quit Program"
    ]
    print_centered_box("Real-time Operations", items)

def handle_navigation_choice(choice: str, router) -> Tuple[bool, Optional[SmartTrashRouter]]:
    """Handle navigation choices, returns (should_exit, new_router)"""
    choice = choice.lower().strip()
    if choice == 'q':
        print("\n")
        print(center_text(apply_style("Thank you for using Smart Trash Routing System!", GREEN)))
        print()
        return True, None
    return False, router

def handle_status_menu(router):
    """Handle status menu operations"""
    while True:
        print_header()
        print_status_menu()
        
        choice = center_input("\nEnter your choice (b to go back, q to quit): ").strip().lower()
        
        if choice == '1':
            view_bins_status(router)
        elif choice == '2':
            view_trucks_status(router)
        elif choice == '3':
            view_high_priority_bins(router)
        elif choice == 'b':
            return False, router
        elif choice == 'q':
            return True, None
        else:
            print(center_text(apply_style("Invalid choice. Please try again.", ORANGE)))
            center_input("Press Enter to continue...")

def handle_route_menu(router):
    """Handle route management operations"""
    while True:
        print_header()
        print_route_menu()
        
        choice = center_input("\nEnter your choice (b to go back, q to quit): ").strip().lower()
        
        if choice == '1':
            calculate_and_view_routes(router)
        elif choice == '2':
            view_current_routes(router)
        elif choice == '3':
            view_route_statistics(router)
        elif choice == 'b':
            return False, router
        elif choice == 'q':
            return True, None
        else:
            print(center_text(apply_style("Invalid choice. Please try again.", ORANGE)))
            center_input("Press Enter to continue...")

def handle_network_menu(router):
    """Handle network analysis operations"""
    while True:
        print_header()
        print_network_menu()
        
        choice = center_input("\nEnter your choice (b to go back, q to quit): ").strip().lower()
        
        if choice == '1':
            view_complete_network(router)
        elif choice == '2':
            view_clusters(router)
        elif choice == '3':
            view_minimum_spanning_tree(router)
        elif choice == 'b':
            return False, router
        elif choice == 'q':
            return True, None
        else:
            print(center_text(apply_style("Invalid choice. Please try again.", ORANGE)))
            center_input("Press Enter to continue...")

def handle_operations_menu(router):
    """Handle real-time operations"""
    while True:
        print_header()
        print_operations_menu()
        
        choice = center_input("\nEnter your choice (b to go back, q to quit): ").strip().lower()
        
        if choice == '1':
            update_bin_fill_level(router)
        elif choice == '2':
            simulate_updates(router)
        elif choice == '3':
            view_recent_updates(router)
        elif choice == 'b':
            return False, router
        elif choice == 'q':
            return True, None
        else:
            print(center_text(apply_style("Invalid choice. Please try again.", ORANGE)))
            center_input("Press Enter to continue...")

def view_current_routes(router):
    """Display current routes for all trucks"""
    print_centered_box("Current Routes")
    
    for truck in router.trucks:
        if truck.route:
            print(center_text(f"\nTruck {truck.id} Route:"))
            print(center_text(" -> ".join(truck.route)))
        else:
            print(center_text(f"\nTruck {truck.id}: No route assigned"))
    
    center_input("\nPress Enter to continue...")

def view_route_statistics(router):
    """Display statistics for current routes"""
    print_centered_box("Route Statistics")
    
    for truck in router.trucks:
        if truck.route:
            try:
                # Calculate total distance using shortest paths between consecutive points
                total_distance = 0
                total_load = 0
                
                for i in range(len(truck.route)-1):
                    try:
                        # Use shortest_path_length to find distance between consecutive points
                        # This works even if points aren't directly connected
                        distance = nx.shortest_path_length(
                            router.graph,
                            truck.route[i],
                            truck.route[i+1],
                            weight='weight'
                        )
                        total_distance += distance
                    except nx.NetworkXNoPath:
                        print(center_text(f"Warning: No path found between {truck.route[i]} and {truck.route[i+1]}"))
                        continue
                
                # Calculate total load (excluding start location)
                for bin_id in truck.route[1:]:  # Exclude start location
                    if bin_id in router.bins:
                        total_load += router.bins[bin_id].fill_level
                
                print(center_text(f"\nTruck {truck.id}:"))
                print(center_text(f"Total Distance: {total_distance:.2f} units"))
                print(center_text(f"Total Load: {total_load:.2f}/{truck.capacity:.2f}"))
                print(center_text(f"Bins Visited: {len(truck.route)-1}"))
                
            except Exception as e:
                print(center_text(f"\nTruck {truck.id}: Error calculating statistics"))
                print(center_text(f"Error: {str(e)}"))
        else:
            print(center_text(f"\nTruck {truck.id}: No route statistics available"))
    
    center_input("\nPress Enter to continue...")

def view_mst(router):
    """Display the Minimum Spanning Tree"""
    print_centered_box("Minimum Spanning Tree Visualization")
    
    if router.mst:
        # Create a new figure for MST only
        plt.figure(figsize=(12, 8))
        pos = {bin_id: (router.bins[bin_id].longitude, router.bins[bin_id].latitude) 
               for bin_id in router.bins}
        
        # Draw MST edges with weight labels
        nx.draw_networkx_edges(router.mst, pos, edge_color='green', width=2)
        
        # Add edge labels for MST
        edge_labels = {}
        for (u, v) in router.mst.edges():
            edge_labels[(u, v)] = f'{router.mst[u][v]["weight"]:.1f}'
        
        nx.draw_networkx_edge_labels(
            router.mst,
            pos,
            edge_labels=edge_labels,
            font_size=8
        )
        
        # Draw nodes with increased size
        nx.draw_networkx_nodes(
            router.mst,
            pos,
            node_size=700,
            node_color='lightblue'
        )
        
        # Add node labels with fill levels
        node_labels = {
            n: f'{n}\n{router.bins[n].fill_percentage:.0f}%'
            for n in router.mst.nodes()
        }
        nx.draw_networkx_labels(
            router.mst,
            pos,
            labels=node_labels,
            font_size=8,
            font_weight='bold'
        )
        
        plt.title("Minimum Spanning Tree")
        plt.axis('on')
        plt.grid(True)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='green', label='MST Edge'),
            plt.scatter([0], [0], c='lightblue', alpha=1, label='Bin')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.show()
    else:
        print(center_text("MST not available. Please initialize the router first."))
    
    center_input("\nPress Enter to continue...")

def view_recent_updates(router):
    """Display recent bin updates"""
    print_centered_box("Recent Bin Updates")
    
    # Sort bins by last update time
    recent_updates = sorted(
        [bin_obj for bin_obj in router.bins.values() if bin_obj.last_updated],
        key=lambda x: x.last_updated,
        reverse=True
    )[:10]  # Show last 10 updates
    
    if recent_updates:
        print("\n" + center_text("Last 10 Updates:"))
        print(center_text("Bin ID\tFill Level\tUpdated At"))
        print(center_text("-" * 40))
        
        for bin_obj in recent_updates:
            update_time = bin_obj.last_updated.strftime("%Y-%m-%d %H:%M:%S")
            print(center_text(
                f"{bin_obj.id}\t{bin_obj.fill_percentage:.1f}%\t{update_time}"
            ))
    else:
        print(center_text("\nNo recent updates available."))
    
    center_input("\nPress Enter to continue...")

def view_bins_status(router):
    """Display status of all bins"""
    print_centered_box("Bins Status")
    
    # Print header
    headers = ["ID", "Fill %", "Priority", "Last Updated"]
    header_line = "\t".join(headers)
    print(center_text(apply_style(header_line, FROST)))
    print(center_text(apply_style("-" * len(header_line), STEEL)))
    
    # Print each bin's status
    for bin_id, bin_obj in router.bins.items():
        priority = apply_style("HIGH", RED) if bin_obj.is_high_priority else apply_style("Normal", GREEN)
        last_updated = bin_obj.last_updated.strftime("%Y-%m-%d %H:%M:%S") if bin_obj.last_updated else "N/A"
        status_line = f"{apply_style(bin_id, YELLOW)}\t{apply_style(f'{bin_obj.fill_percentage:.1f}%', FROST)}\t{priority}\t{apply_style(last_updated, STEEL)}"
        print(center_text(status_line))
    
    center_input("\nPress Enter to continue...")

def view_trucks_status(router):
    """Display status of all trucks"""
    print_centered_box("Trucks Status")
    
    # Print header
    headers = ["ID", "Capacity", "Current Load", "Available", "Current Location"]
    header_line = "\t".join(headers)
    print(center_text(apply_style(header_line, FROST)))
    print(center_text(apply_style("-" * len(header_line), STEEL)))
    
    # Print each truck's status
    for truck in router.trucks:
        status_line = (
            f"{apply_style(truck.id, YELLOW)}\t"
            f"{apply_style(f'{truck.capacity:.1f}', FROST)}\t"
            f"{apply_style(f'{truck.current_load:.1f}', ORANGE)}\t"
            f"{apply_style(f'{truck.available_capacity:.1f}', GREEN)}\t"
            f"{apply_style(truck.current_location, PURPLE)}"
        )
        print(center_text(status_line))
    
    center_input("\nPress Enter to continue...")

def update_bin_fill_level(router):
    """Update the fill level of a specific bin"""
    print_centered_box("Update Bin Fill Level")
    
    # Get bin ID
    print(center_text(apply_style("Available Bins:", FROST)))
    print(center_text(apply_style(", ".join(router.bins.keys()), YELLOW)))
    bin_id = center_input("\nEnter bin ID: ").strip()
    
    if bin_id not in router.bins:
        print(center_text(apply_style("Error: Bin ID not found!", RED)))
        center_input("Press Enter to continue...")
        return
    
    try:
        new_level = float(center_input("Enter new fill level (0-100): "))
        if not 0 <= new_level <= 100:
            raise ValueError("Fill level must be between 0 and 100")
        
        router.update_bin_status(bin_id, new_level)
        print(center_text(apply_style(f"Successfully updated bin {bin_id} fill level to {new_level}%", GREEN)))
    except ValueError as e:
        print(center_text(apply_style(f"Error: {e}", RED)))
    
    center_input("Press Enter to continue...")

def view_clusters(router):
    """Display identified clusters"""
    print_centered_box("Bin Clusters")
    
    if not router.clusters:
        print(center_text("No clusters identified yet."))
    else:
        for i, cluster in enumerate(router.clusters, 1):
            print(center_text(f"Cluster {i} ({len(cluster)} bins):"))
            print(center_text(", ".join(cluster)))
            print()  # Empty line between clusters
    
    center_input("Press Enter to continue...")

def view_high_priority_bins(router):
    """Display high priority bins"""
    print_centered_box("High Priority Bins")
    
    # Print header
    headers = ["ID", "Fill %", "Last Updated"]
    header_line = "\t".join(headers)
    print(center_text(header_line))
    print(center_text("-" * len(header_line)))
    
    high_priority = [bin_obj for bin_obj in router.bins.values() if bin_obj.is_high_priority]
    if not high_priority:
        print(center_text("No high priority bins found."))
    else:
        for bin_obj in high_priority:
            last_updated = bin_obj.last_updated.strftime("%Y-%m-%d %H:%M:%S") if bin_obj.last_updated else "N/A"
            status_line = f"{bin_obj.id}\t{bin_obj.fill_percentage:.1f}%\t{last_updated}"
            print(center_text(status_line))
    
    center_input("\nPress Enter to continue...")

def simulate_updates(router):
    """Simulate real-time updates to random bins"""
    import random
    print_centered_box("Simulating Real-time Updates")
    
    try:
        num_updates = int(center_input("Enter number of random updates to simulate: "))
        bin_ids = list(router.bins.keys())
        
        for i in range(num_updates):
            bin_id = random.choice(bin_ids)
            new_level = random.uniform(0, 100)
            old_level = router.bins[bin_id].fill_level
            
            print(center_text(f"\nUpdate {i+1}/{num_updates}:"))
            print(center_text(f"Bin {bin_id}: {old_level:.1f}% -> {new_level:.1f}%"))
            
            router.update_bin_status(bin_id, new_level)
            center_input("Press Enter for next update...")
    except ValueError:
        print(center_text("Error: Please enter a valid number."))
        center_input("Press Enter to continue...")

def calculate_and_view_routes(router):
    """Calculate and display routes for all trucks"""
    print_centered_box("Calculating Routes")
    
    try:
        print(center_text(apply_style("Starting route calculation...", FROST)))
        assignments = router.assign_bins_to_trucks()
        
        if not assignments:
            print(center_text(apply_style("No bin assignments found.", ORANGE)))
            center_input("\nPress Enter to continue...")
            return
            
        print(center_text(apply_style(f"Found assignments for {len(assignments)} trucks", GREEN)))
        
        for truck_id, assigned_bins in assignments.items():
            print(center_text(apply_style(f"\nProcessing route for {truck_id}:", PURPLE)))
            print(center_text(apply_style(f"Assigned bins: {', '.join(assigned_bins)}", YELLOW)))
            
            try:
                truck = next(t for t in router.trucks if t.id == truck_id)
                print(center_text(apply_style(f"Starting from location: {truck.current_location}", FROST)))
                
                route = router.find_optimal_route(truck, assigned_bins)
                truck.route = route
                
                if route:
                    print(center_text(apply_style(f"Route found: {' -> '.join(route)}", GREEN)))
                    view_route = center_input("\nView route visualization? (y/n): ").lower().strip()
                    if view_route == 'y':
                        router.visualize_route(route, f"Route for {truck_id}")
                else:
                    print(center_text(apply_style("No valid route found for this truck.", ORANGE)))
            except Exception as e:
                print(center_text(apply_style(f"Error processing route for {truck_id}: {str(e)}", RED)))
        
    except Exception as e:
        print(center_text(apply_style(f"Error calculating routes: {str(e)}", RED)))
    
    center_input("\nPress Enter to continue...")

def select_test_data():
    menu_items = [
        "1. Basic Test        (06 bins, 01 truck)",
        "2. Constraint Test   (12 bins, 02 trucks)",
        "3. Optimization Test (24 bins, 03 trucks)",
        "q. Quit Program"
    ]
    
    # Find the longest item length for centering calculation
    max_length = max(len(strip_ansi(item)) for item in menu_items)
    width, _ = get_terminal_size()
    
    # Calculate center padding (no left offset)
    base_padding = (width - max_length) // 2
    
    # Print menu items with consistent left-aligned offset from center
    print()  # Add spacing before menu
    for item in menu_items:
        # Add extra right padding for "q. Quit Program" to maintain alignment
        if item.startswith('q.'):
            item_padding = base_padding + 15
        else:
            item_padding = base_padding
        print(' ' * item_padding + apply_style(item, TEXT_LIGHTER))
    print()  # Add spacing after menu
    
    while True:
        choice = center_input("\nEnter your choice: ").strip().lower()
        
        if choice == '1':
            return 'data/basic_test'
        elif choice == '2':
            return 'data/constraint_test'
        elif choice == '3':
            return 'data/optimization_test'
        elif choice == 'q':
            return None
        else:
            print(center_text(apply_style("Invalid choice. Please try again.", ORANGE)))
            center_input("Press Enter to continue...")

def initialize_router(data_path: str) -> Optional[SmartTrashRouter]:
    """Initialize router with selected test data"""
    try:
        router = SmartTrashRouter(
            f'{data_path}/bins.csv',
            f'{data_path}/roads.csv',
            f'{data_path}/trucks.csv'
        )
        print(center_text("\nSuccessfully loaded test data!"))
        print(center_text(f"Loaded {len(router.bins)} bins and {len(router.trucks)} trucks"))
        center_input("\nPress Enter to continue...")
        return router
    except Exception as e:
        print(center_text(f"\nError loading test data: {str(e)}"))
        center_input("\nPress Enter to continue...")
        return None

def main():
    """Main application entry point."""
    # Initialize terminal with default styling
    init_terminal()
    
    router = None
    
    while True:
        print_header()
        
        if router is None:
            print(center_text(apply_style("Warning: No data loaded!\n", RED)))
            print(center_text(apply_style("Please select test data to continue...\n", STEEL)))
            
            data_path = select_test_data()
            if data_path is None:
                print("\n")
                print_centered_box("Thank you for using Smart Trash Routing System!")
                print()
                break
                
            router = initialize_router(data_path)
            if router is None:
                continue
            
        print_main_menu()
        choice = center_input("\nEnter your choice: ").strip().lower()
        
        should_exit = False
        if choice == '1':
            should_exit, router = handle_status_menu(router)
        elif choice == '2':
            should_exit, router = handle_route_menu(router)
        elif choice == '3':
            should_exit, router = handle_network_menu(router)
        elif choice == '4':
            should_exit, router = handle_operations_menu(router)
        elif choice == '5':
            router = None
            continue
        elif choice == 'q':
            print("\n")
            print_centered_box("Thank you for using Smart Trash Routing System!")
            print()
            break
        else:
            print(center_text(apply_style("Invalid choice. Please try again.", ORANGE)))
            center_input("Press Enter to continue...")
        
        if should_exit:
            print("\n")
            print_centered_box("Thank you for using Smart Trash Routing System!")
            print()
            break

if __name__ == "__main__":
    main() 