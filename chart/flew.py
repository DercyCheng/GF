import matplotlib.pyplot as plt
import networkx as nx
import os
import matplotlib as mpl
import matplotlib

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Set scientific publication-oriented style
# Check matplotlib version and use appropriate style
if matplotlib.__version__ >= '3.6':
    # For newer matplotlib versions
    plt.style.use('seaborn-v0_8-whitegrid')
else:
    # Fallback to a basic style for any version
    plt.style.use('default')
    plt.grid(True)

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['figure.titlesize'] = 16

def create_dataset_flowchart():
    """Create flowchart for dataset.py processing pipeline with academic style"""
    G = nx.DiGraph()
    
    # Simplified node structure focused on key methodology steps
    nodes = [
        "Raw Data", 
        "Data Processing",
        "SGD Processing",
        "SNV Normalization",
        "MSC Correction", 
        "DWT Analysis",
        "Dataset Construction"
    ]
    
    # Create nodes with hierarchical positions for better layout
    node_positions = {
        "Raw Data": (0, 2),
        "Data Processing": (1, 2),
        "SGD Processing": (2, 3),
        "SNV Normalization": (2, 2),
        "MSC Correction": (2, 1), 
        "DWT Analysis": (2, 0),
        "Dataset Construction": (3, 2)
    }
    
    for node in nodes:
        G.add_node(node)
    
    # Add edges representing scientific workflow
    edges = [
        ("Raw Data", "Data Processing"),
        ("Data Processing", "SGD Processing"),
        ("Data Processing", "SNV Normalization"),
        ("Data Processing", "MSC Correction"),
        ("Data Processing", "DWT Analysis"),
        ("SGD Processing", "Dataset Construction"),
        ("SNV Normalization", "Dataset Construction"),
        ("MSC Correction", "Dataset Construction"),
        ("DWT Analysis", "Dataset Construction")
    ]
    
    G.add_edges_from(edges)
    
    # Create the plot with scientific aesthetics
    plt.figure(figsize=(8, 4), dpi=300)
    nx.draw_networkx(G, pos=node_positions, with_labels=True, 
                    node_size=2000, node_color="#e6f2ff", 
                    font_size=9, font_weight="bold",
                    edge_color="#555555", arrows=True,
                    arrowsize=15, arrowstyle='-|>')
    
    ensure_dir("./datasets/flowcharts")
    plt.title("Spectral Dataset Processing Methodology", fontweight='bold', pad=20)
    plt.tight_layout()
    plt.axis('off')
    plt.savefig("./datasets/flowcharts/dataset_flowchart.png", bbox_inches="tight", dpi=300)
    plt.close()

def create_train_flowchart():
    """Create flowchart for train.py execution flow with scientific aesthetics"""
    G = nx.DiGraph()
    
    # More concise, research-focused methodology nodes
    nodes = [
        "Data Acquisition",
        "Spectral Preprocessing", 
        "Dimensionality\nReduction",
        "Model Architecture\nSelection",
        "Cross-Validation",
        "Training Phase",
        "Performance\nEvaluation",
        "Hyperparameter\nOptimization",
        "Results Analysis"
    ]
    
    # Hierarchical node positioning for clearer scientific workflow
    node_positions = {
        "Data Acquisition": (0, 2),
        "Spectral Preprocessing": (1, 2),
        "Dimensionality\nReduction": (2, 2),
        "Model Architecture\nSelection": (3, 2),
        "Cross-Validation": (4, 2),
        "Training Phase": (5, 2),
        "Performance\nEvaluation": (6, 2),
        "Hyperparameter\nOptimization": (6, 0.5),
        "Results Analysis": (7, 2)
    }
    
    for node in nodes:
        G.add_node(node)
    
    edges = [
        ("Data Acquisition", "Spectral Preprocessing"),
        ("Spectral Preprocessing", "Dimensionality\nReduction"),
        ("Dimensionality\nReduction", "Model Architecture\nSelection"),
        ("Model Architecture\nSelection", "Cross-Validation"),
        ("Cross-Validation", "Training Phase"),
        ("Training Phase", "Performance\nEvaluation"),
        ("Performance\nEvaluation", "Hyperparameter\nOptimization"),
        ("Hyperparameter\nOptimization", "Training Phase"),
        ("Performance\nEvaluation", "Results Analysis")
    ]
    
    G.add_edges_from(edges)
    
    # Create the plot with scientific aesthetics
    plt.figure(figsize=(10, 3.5), dpi=300)
    nx.draw_networkx(G, pos=node_positions, with_labels=True, 
                     node_size=2200, node_color="#f0f9e8", 
                     font_size=8, font_weight="bold",
                     edge_color="#555555", arrows=True,
                     arrowsize=15, arrowstyle='-|>')
    
    ensure_dir("./datasets/flowcharts")
    plt.title("Soil Properties Prediction Methodology", fontweight='bold', pad=20)
    plt.tight_layout()
    plt.axis('off')
    plt.savefig("./datasets/flowcharts/train_flowchart.png", bbox_inches="tight", dpi=300)
    plt.close()

def create_dcnn_flowchart():
    """Create flowchart for DCNN.py model architecture with scientific visualization"""
    G = nx.DiGraph()
    
    # Architecture components with technical terminology
    nodes = [
        "Input Tensor\n(b × 1 × w)",
        "Initial Conv1D\n(64 filters)",
        "Residual Block\n(64→128)",
        "Attention Module\n(SE/ECA/CBAM)",
        "Residual Block\n(128→256)",
        "Attention Module\n(SE/ECA/CBAM)",
        "Residual Block\n(256→512)",
        "Attention Module\n(SE/ECA/CBAM)",
        "Global Pooling",
        "Dense Layer\n(512→256)",
        "Output Layer\n(256→1)"
    ]
    
    # Precise positioning for model architecture visualization
    node_positions = {
        "Input Tensor\n(b × 1 × w)": (0, 5),
        "Initial Conv1D\n(64 filters)": (1, 5),
        "Residual Block\n(64→128)": (2, 5),
        "Attention Module\n(SE/ECA/CBAM)": (3, 5),
        "Residual Block\n(128→256)": (4, 5),
        "Attention Module\n(SE/ECA/CBAM)": (5, 5),
        "Residual Block\n(256→512)": (6, 5),
        "Attention Module\n(SE/ECA/CBAM)": (7, 5),
        "Global Pooling": (8, 5),
        "Dense Layer\n(512→256)": (9, 5),
        "Output Layer\n(256→1)": (10, 5)
    }
    
    for node in nodes:
        G.add_node(node)
    
    # Sequential architecture connections
    edges = [
        ("Input Tensor\n(b × 1 × w)", "Initial Conv1D\n(64 filters)"),
        ("Initial Conv1D\n(64 filters)", "Residual Block\n(64→128)"),
        ("Residual Block\n(64→128)", "Attention Module\n(SE/ECA/CBAM)"),
        ("Attention Module\n(SE/ECA/CBAM)", "Residual Block\n(128→256)"),
        ("Residual Block\n(128→256)", "Attention Module\n(SE/ECA/CBAM)"),
        ("Attention Module\n(SE/ECA/CBAM)", "Residual Block\n(256→512)"),
        ("Residual Block\n(256→512)", "Attention Module\n(SE/ECA/CBAM)"),
        ("Attention Module\n(SE/ECA/CBAM)", "Global Pooling"),
        ("Global Pooling", "Dense Layer\n(512→256)"),
        ("Dense Layer\n(512→256)", "Output Layer\n(256→1)")
    ]
    
    G.add_edges_from(edges)
    
    # Create the plot with technical diagram aesthetics
    plt.figure(figsize=(12, 4), dpi=300)
    nx.draw_networkx(G, pos=node_positions, with_labels=True, 
                     node_size=2500, node_color="#f1f6ff", 
                     font_size=8, font_weight="bold",
                     edge_color="#333333", arrows=True,
                     arrowsize=15, arrowstyle='-|>')
    
    ensure_dir("./datasets/flowcharts")
    plt.title("Attention-Enhanced DCNN Architecture for Spectral Analysis", fontweight='bold', pad=20)
    plt.tight_layout()
    plt.axis('off')
    plt.savefig("./datasets/flowcharts/dcnn_flowchart.png", bbox_inches="tight", dpi=300)
    plt.close()

def create_utils_flowchart():
    """Create flowchart for utils.py functions with categorical organization"""
    # Use directed graph with clusters
    G = nx.DiGraph()
    
    # Group utility functions by categories for better organization
    function_categories = {
        "Data Processing": [
            "load_data()",
            "preprocess_data()", 
            "augment_data()"
        ],
        "Visualization": [
            "plot_results()",
            "plot_loss_curve()",
            "plot_spectral_curves()",
            "plot_correlation_matrix()"
        ],
        "Model Evaluation": [
            "evaluate_model()",
            "plot_regression_diagnostics()",
            "plot_feature_importance()"
        ],
        "Explainability": [
            "shap_analysis()",
            "lime_analysis()"
        ]
    }
    
    # Create hierarchical layout
    node_positions = {}
    
    # Position categories and their functions
    x_pos = 0
    for category, functions in function_categories.items():
        # Add category node
        G.add_node(category)
        node_positions[category] = (x_pos, 4)
        
        # Add function nodes and connect to category
        y_pos = 1.5
        for function in functions:
            G.add_node(function)
            node_positions[function] = (x_pos, y_pos)
            G.add_edge(category, function)
            y_pos -= 0.4
        
        x_pos += 4
    
    # Create the plot with academic aesthetics
    plt.figure(figsize=(12, 3.5), dpi=300)
    
    # Draw nodes with different colors for categories and functions
    category_nodes = list(function_categories.keys())
    function_nodes = [f for funcs in function_categories.values() for f in funcs]
    
    nx.draw_networkx_nodes(G, pos=node_positions, 
                          nodelist=category_nodes,
                          node_size=3000, 
                          node_color="#c6dbef")
    
    nx.draw_networkx_nodes(G, pos=node_positions, 
                          nodelist=function_nodes,
                          node_size=2000, 
                          node_color="#e6f2ff")
    
    nx.draw_networkx_edges(G, pos=node_positions, 
                          arrows=True, 
                          arrowsize=15,
                          arrowstyle='-|>',
                          edge_color="#555555")
    
    nx.draw_networkx_labels(G, pos=node_positions, 
                           font_size=9,
                           font_weight="bold")
    
    ensure_dir("./datasets/flowcharts")
    plt.title("Utility Functions for Spectral Analysis Framework", fontweight='bold', pad=20)
    plt.tight_layout()
    plt.axis('off')
    plt.savefig("./datasets/flowcharts/utils_flowchart.png", bbox_inches="tight", dpi=300)
    plt.close()

def create_integrated_flowchart():
    """Create integrated framework diagram showing relationships between components"""
    G = nx.DiGraph()
    
    # Create a clear three-layer architecture
    component_layers = {
        "Data Layer": [
            "Spectral\nPreprocessing",
            "Feature\nEngineering"
        ],
        "Model Layer": [
            "1D-CNN\nModels",
            "Attention\nMechanisms",
            "ML\nAlgorithms"
        ],
        "Analysis Layer": [
            "Performance\nMetrics",
            "Explainability\nTools",
            "Visualization"
        ]
    }
    
    # Create structured positions for components
    node_positions = {}
    
    # Position layers and their components
    y_pos = 4
    for layer, components in component_layers.items():
        # Add layer node
        G.add_node(layer)
        node_positions[layer] = (3, y_pos)
        
        # Add component nodes
        x_pos = 0
        for component in components:
            G.add_node(component)
            node_positions[component] = (x_pos, y_pos - 1.5)
            G.add_edge(layer, component)
            x_pos += 3
        
        # Connect to next layer
        if layer != "Analysis Layer":
            G.add_edge(layer, list(component_layers.keys())[list(component_layers.keys()).index(layer) + 1])
        
        y_pos -= 2
    
    # Add specific connections between components
    G.add_edge("Spectral\nPreprocessing", "Feature\nEngineering")
    G.add_edge("Feature\nEngineering", "1D-CNN\nModels")
    G.add_edge("Feature\nEngineering", "ML\nAlgorithms")
    G.add_edge("1D-CNN\nModels", "Attention\nMechanisms")
    G.add_edge("1D-CNN\nModels", "Performance\nMetrics")
    G.add_edge("ML\nAlgorithms", "Performance\nMetrics")
    G.add_edge("Performance\nMetrics", "Visualization")
    G.add_edge("Performance\nMetrics", "Explainability\nTools")
    
    # Create the plot with academic aesthetics
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Draw nodes with different colors for layers and components
    layer_nodes = list(component_layers.keys())
    component_nodes = [c for comps in component_layers.values() for c in comps]
    
    nx.draw_networkx_nodes(G, pos=node_positions, 
                          nodelist=layer_nodes,
                          node_size=3500, 
                          node_color="#9ecae1",
                          alpha=0.7)
    
    nx.draw_networkx_nodes(G, pos=node_positions, 
                          nodelist=component_nodes,
                          node_size=2500, 
                          node_color="#deebf7")
    
    nx.draw_networkx_edges(G, pos=node_positions, 
                          arrows=True, 
                          arrowsize=20,
                          arrowstyle='-|>',
                          edge_color="#555555")
    
    nx.draw_networkx_labels(G, pos=node_positions, 
                           font_size=10,
                           font_weight="bold")
    
    ensure_dir("./datasets/flowcharts")
    plt.title("Integrated Framework for Soil Properties Prediction", fontweight='bold', pad=20)
    plt.tight_layout()
    plt.axis('off')
    plt.savefig("./datasets/flowcharts/integrated_flowchart.png", bbox_inches="tight", dpi=300)
    plt.close()

def create_comprehensive_workflow():
    """Create a comprehensive flowchart with enhanced scientific color scheme and academic styling"""
    G = nx.DiGraph()
    
    # Define all major components with reduced vertical spacing
    components = {
        # Data processing components - Reduced vertical spacing
        "Raw Data": {"layer": "Data", "position": (0, 5)},
        "Data Processing": {"layer": "Data", "position": (2.5, 5)},
        "SGD Processing": {"layer": "Processing", "position": (5, 8)},
        "SNV Normalization": {"layer": "Processing", "position": (5, 5)},
        "MSC Correction": {"layer": "Processing", "position": (5, 2)},
        "DWT Analysis": {"layer": "Processing", "position": (5, -1)},
        
        # Model components - Maintained horizontal spacing with reduced vertical
        "Model Architecture": {"layer": "Model", "position": (10, 5)},
        "Residual Blocks": {"layer": "Model", "position": (13, 8)},
        "Attention\nMechanisms": {"layer": "Model", "position": (13, 5)},
        "Convolutional\nLayers": {"layer": "Model", "position": (13, 2)},
        "Dense Layers": {"layer": "Model", "position": (13, -1)},
        
        # Training components - Reduced vertical space
        "Training Phase": {"layer": "Training", "position": (18, 5)},
        "Cross-Validation": {"layer": "Training", "position": (21, 8)},
        "Hyperparameter\nOptimization": {"layer": "Training", "position": (21, 5)},
        "Loss Function": {"layer": "Training", "position": (21, 2)},
        
        # Evaluation components - Reduced vertical spacing
        "Performance\nEvaluation": {"layer": "Evaluation", "position": (26, 5)},
        "Metrics\nCalculation": {"layer": "Evaluation", "position": (29, 8)},
        "Visualization": {"layer": "Evaluation", "position": (29, 5)},
        "Explainability": {"layer": "Evaluation", "position": (29, 2)},
        
        # Output - Adjusted position
        "Prediction\nResults": {"layer": "Output", "position": (34, 5)}
    }
    
    # Add all nodes to the graph
    for component, attrs in components.items():
        G.add_node(component, **attrs)
    
    # Define edges between components
    edges = [
        # Data flow
        ("Raw Data", "Data Processing"),
        ("Data Processing", "SGD Processing"),
        ("Data Processing", "SNV Normalization"),
        ("Data Processing", "MSC Correction"),
        ("Data Processing", "DWT Analysis"),
        ("SGD Processing", "Model Architecture"),
        ("SNV Normalization", "Model Architecture"),
        ("MSC Correction", "Model Architecture"),
        ("DWT Analysis", "Model Architecture"),
        
        # Model architecture flow
        ("Model Architecture", "Residual Blocks"),
        ("Model Architecture", "Attention\nMechanisms"),
        ("Model Architecture", "Convolutional\nLayers"),
        ("Model Architecture", "Dense Layers"),
        ("Residual Blocks", "Training Phase"),
        ("Attention\nMechanisms", "Training Phase"),
        ("Convolutional\nLayers", "Training Phase"),
        ("Dense Layers", "Training Phase"),
        
        # Training flow
        ("Training Phase", "Cross-Validation"),
        ("Training Phase", "Hyperparameter\nOptimization"),
        ("Training Phase", "Loss Function"),
        ("Cross-Validation", "Performance\nEvaluation"),
        ("Hyperparameter\nOptimization", "Performance\nEvaluation"),
        ("Loss Function", "Performance\nEvaluation"),
        
        # Evaluation flow
        ("Performance\nEvaluation", "Metrics\nCalculation"),
        ("Performance\nEvaluation", "Visualization"),
        ("Performance\nEvaluation", "Explainability"),
        ("Metrics\nCalculation", "Prediction\nResults"),
        ("Visualization", "Prediction\nResults"),
        ("Explainability", "Prediction\nResults"),
        
        # Feedback loops
        ("Hyperparameter\nOptimization", "Model Architecture"),
        ("Performance\nEvaluation", "Training Phase")
    ]
    
    G.add_edges_from(edges)
    
    # Enhanced scientific color scheme inspired by academic publications
    layer_colors = {
        "Data": "#D5E8F0",         # Soft blue (data collection)
        "Processing": "#B9DFC9",    # Mint green (preprocessing)
        "Model": "#F0E6D2",         # Warm beige (model development)
        "Training": "#E3D2EF",      # Soft purple (training phase)
        "Evaluation": "#F9D5CA",    # Peach (evaluation)
        "Output": "#FFE5B4"         # Light amber (results)
    }
    
    # Scientific edge styling
    edge_colors = {
        "forward": "#486B77",       # Slate blue-gray
        "feedback": "#A4686B"       # Muted burgundy
    }
    
    # Node border colors - deeper version of fill color for scientific graphs
    border_colors = {
        "Data": "#7BAEC1",          # Deeper blue
        "Processing": "#74A989",    # Deeper green
        "Model": "#C9B18A",         # Deeper beige
        "Training": "#B095C3",      # Deeper purple
        "Evaluation": "#C9937C",    # Deeper peach
        "Output": "#D9B668"         # Deeper amber
    }
    
    # Set up figure with scientific aesthetics
    plt.figure(figsize=(24, 10), dpi=300)
    plt.tight_layout(pad=3.0)
    
    # Scientific graph background
    ax = plt.gca()
    ax.set_facecolor("#FCFCFC")    # Nearly white background common in publications
    
    # Add subtle grid for measurement reference (common in scientific graphs)
    plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.3, zorder=-1000)
    
    # Get position dictionary for nodes
    pos = {n: components[n]['position'] for n in G.nodes()}
    
    # Draw background shading for layer groups with scientific styling
    for layer, color in layer_colors.items():
        nodes = [n for n, d in G.nodes(data=True) if d.get('layer') == layer]
        if nodes:
            positions = [components[n]['position'] for n in nodes]
            x_vals = [p[0] for p in positions]
            y_vals = [p[1] for p in positions]
            min_x = min(x_vals) - 2.5
            max_x = max(x_vals) + 2.5
            min_y = min(y_vals) - 2.0
            max_y = max(y_vals) + 2.5
            
            # Add subtle gradient effect common in scientific visualizations
            rect = plt.Rectangle((min_x, min_y), max_x-min_x, max_y-min_y,
                                fill=True, color=color, alpha=0.15,  # More subtle
                                edgecolor=border_colors[layer], linewidth=1.2,
                                linestyle='-', zorder=-100)
            plt.gca().add_patch(rect)
    
    # Draw nodes with enhanced academic styling
    for layer, color in layer_colors.items():
        nodes = [n for n, d in G.nodes(data=True) if d.get('layer') == layer]
        
        # Shadow for depth - used in professional scientific presentations
        nx.draw_networkx_nodes(G, 
                            pos={n: (pos[n][0]+0.15, pos[n][1]-0.15) for n in nodes},
                            nodelist=nodes,
                            node_size=3600, 
                            node_color='#E0E0E0',  # Light gray shadow
                            alpha=0.25,
                            edgecolors=None)
        
        # Main nodes with scientific coloring
        nx.draw_networkx_nodes(G, 
                            pos=pos,
                            nodelist=nodes,
                            node_size=3600,
                            node_color=color,
                            alpha=0.85,  # Slightly transparent like in scientific graphs
                            edgecolors=border_colors[layer],
                            linewidths=1.8)
    
    # Edge drawing with scientific styling - clear directed pathways
    for edge in G.edges():
        src, dst = edge
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        
        # Determine edge type (feedback or forward)
        is_feedback = False
        if edge in [("Hyperparameter\nOptimization", "Model Architecture"), 
                   ("Performance\nEvaluation", "Training Phase")]:
            is_feedback = True
            edge_color = edge_colors["feedback"]
            style = 'dashed'
        else:
            edge_color = edge_colors["forward"]
            style = 'solid'
        
        # Scientific edge routing - optimize for clarity
        rad = 0.0
        
        # Vertical connection adjustment
        if abs(x1 - x2) < 0.1:
            rad = 0.1
        # Horizontal connection adjustment
        elif abs(y1 - y2) < 0.1:
            rad = 0.0
        # Diagonal connection adjustment
        else:
            if x1 > x2:  # If going backwards
                rad = -0.3
            elif is_feedback:  # If feedback loop
                rad = 0.4
            else:  # Normal forward flow
                rad = 0.2
        
        # Scientific arrow styling - like in research diagrams
        nx.draw_networkx_edges(G, 
                            pos,
                            edgelist=[edge],
                            arrows=True,
                            arrowsize=28,
                            arrowstyle='simple',  # Clean scientific style
                            edge_color=edge_color,
                            width=1.6,
                            alpha=0.9,
                            style=style,
                            connectionstyle=f'arc3,rad={rad}')
    
    # Add labels with research-style formatting
    for node, (x, y) in pos.items():
        layer = next((l for l, nodes in [(l, [n for n, d in G.nodes(data=True) if d.get('layer') == l]) 
                             for l in layer_colors.keys()] if node in nodes), None)
        
        # Text with scientific-style label boxes
        plt.text(x, y, node, 
                fontsize=11,
                fontweight='bold',
                ha='center', va='center',
                color='#202020',  # Dark gray text common in publications
                bbox=dict(boxstyle="round,pad=0.6",
                        facecolor="white",
                        edgecolor=border_colors[layer] if layer else "gray",
                        alpha=0.95),
                zorder=100)
    
    # Add layer titles with academic journal styling
    for layer, color in layer_colors.items():
        nodes = [n for n, d in G.nodes(data=True) if d.get('layer') == layer]
        if nodes:
            x = sum(components[n]['position'][0] for n in nodes) / len(nodes)
            y_max = max(components[n]['position'][1] for n in nodes) + 2.0
            
            plt.text(x, y_max, 
                    f"{layer} Layer", 
                    fontsize=16, 
                    fontweight='bold',
                    ha='center',
                    va='center',
                    color='#202020',
                    bbox=dict(facecolor=color, 
                            alpha=0.6,  
                            edgecolor=border_colors[layer],
                            boxstyle='round,pad=0.5',
                            mutation_scale=1.1))
    
    # Scientific paper-style title
    plt.text(17, 12, "Comprehensive Soil Spectral Analysis Framework",
            fontsize=22, 
            fontweight='bold',
            ha='center',
            color='#202020',
            bbox=dict(facecolor='white', 
                     alpha=0.9,
                     edgecolor='#707070',  # Standard in scientific papers
                     boxstyle='round,pad=0.8'))
    
    # Add scientific legend - like in research publications
    legend_elements = [
        plt.Line2D([0], [0], color=edge_colors["forward"], lw=3, label='Forward Process Flow'),
        plt.Line2D([0], [0], color=edge_colors["feedback"], lw=3, linestyle='--', label='Optimization Feedback Loop')
    ]
    legend = plt.legend(handles=legend_elements, loc='upper right', fontsize=14, 
                       frameon=True, framealpha=0.95, edgecolor='#707070')
    
    ensure_dir("./datasets/flowcharts")
    plt.axis('off')
    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.03)
    plt.savefig("./datasets/flowcharts/comprehensive_workflow.png",
              bbox_inches="tight", 
              dpi=300,
              facecolor='#FCFCFC')
    plt.close()

def main():
    """Generate all flowcharts with publication-quality styling"""
    ensure_dir("./datasets/flowcharts")
    
    # Uncomment individual flowcharts if needed separately
    # create_dataset_flowchart()
    # create_train_flowchart()
    # create_dcnn_flowchart()
    # create_utils_flowchart()
    # create_integrated_flowchart()
    
    # Generate the comprehensive workflow diagram
    create_comprehensive_workflow()
    
    print("Comprehensive workflow diagram has been generated in ./datasets/flowcharts/")

if __name__ == "__main__":
    main()
