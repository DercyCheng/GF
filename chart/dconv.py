import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np

def draw_dynamic_conv_diagram():
    # Create figure and axis with further reduced height
    fig, ax = plt.subplots(figsize=(18, 7))  # Reduced height from 9 to 7
    
    # Set background color
    ax.set_facecolor('#f9f9f9')
    
    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Enhanced color palette
    feature_color = '#3498db'  # blue
    kernel_color = '#e74c3c'   # red
    output_color = '#9b59b6'   # purple
    arrow_color = '#2c3e50'    # darker for better contrast
    math_color = '#34495e'     # dark gray
    highlight_color = '#f39c12' # orange
    dilation_colors = ['#e67e22', '#8e44ad', '#16a085']  # More distinct colors
    
    # Title and subtitle with adjusted positioning
    plt.suptitle('Multi-Scale Dilated Dynamic Convolution', fontsize=18, fontweight='bold', y=0.96)
    
    # FURTHER REDUCED VERTICAL SPACING
    top_section_y = 5.0      # Reduced from 6.2
    middle_section_y = 3.2   # Reduced from 3.8
    bottom_section_y = 1.4   # Same as before
    
    # Draw input feature map in left side with reduced height
    input_rect = patches.Rectangle((1.5, middle_section_y - 1.0), 1.5, 2.0, linewidth=2, 
                                  edgecolor=feature_color, facecolor=feature_color, alpha=0.3)
    ax.add_patch(input_rect)
    ax.text(2.25, middle_section_y + 1.25, 'Input Feature Map X', ha='center', fontsize=12, fontweight='bold')
    ax.text(2.25, middle_section_y - 1.2, r'$\mathbf{X} \in \mathbb{R}^{C_{in} \times H \times W}$', 
           ha='center', fontsize=10, color=math_color, fontweight='bold')
    
    # Add grid to input with adjusted spacing
    for i in range(4):
        ax.plot([1.5, 3], [middle_section_y - 1.0 + i*0.5, middle_section_y - 1.0 + i*0.5], 
               color=feature_color, alpha=0.4, linestyle='-', linewidth=0.5)
    for i in range(3):
        ax.plot([1.5 + i*0.5, 1.5 + i*0.5], [middle_section_y - 1.0, middle_section_y + 1.0], 
               color=feature_color, alpha=0.4, linestyle='-', linewidth=0.5)
    
    # Draw the three filters with even more compact spacing
    filter_positions = [
        (3.5, top_section_y, dilation_colors[0], 1),    # Top filter (R=1)
        (3.5, middle_section_y, dilation_colors[1], 2), # Middle filter (R=2)
        (3.5, bottom_section_y, dilation_colors[2], 3)  # Bottom filter (R=3)
    ]
    
    for pos_x, pos_y, color, dilation_rate in filter_positions:
        height = 0.7  # Further reduced height from 0.8
        width = 1.0   # Fixed width
        
        # Draw filter
        filter_rect = patches.Rectangle((pos_x, pos_y - height/2), width, height, 
                                       linewidth=2, edgecolor=color, facecolor=color, alpha=0.3)
        ax.add_patch(filter_rect)
        
        # Add simplified internal grid with reduced size
        grid_size = 0.23  # Reduced from 0.27
        for i in range(int(height/grid_size) + 1):
            ax.plot([pos_x, pos_x + width], 
                   [pos_y - height/2 + i*grid_size, pos_y - height/2 + i*grid_size], 
                   color=color, alpha=0.5, linestyle='-', linewidth=0.5)
        
        for i in range(int(width/grid_size) + 1):
            ax.plot([pos_x + i*grid_size, pos_x + i*grid_size], 
                   [pos_y - height/2, pos_y + height/2], 
                   color=color, alpha=0.5, linestyle='-', linewidth=0.5)
        
        # Label for filter - moved closer to component
        ax.text(pos_x + width/2, pos_y - height/2 - 0.3, 
               f'Filter F{dilation_rate} (R={dilation_rate})', ha='center', fontsize=9, fontweight='bold')
        
        # Mathematical notation for filter - moved closer
        ax.text(pos_x + width/2, pos_y + height/2 + 0.3, 
               f'$\\mathbf{{K}}_{{{dilation_rate}}}$', ha='center', fontsize=9, color=math_color)
        
        # Draw output feature map - same height as filter
        output_x = 5.5
        output_rect = patches.Rectangle((output_x, pos_y - height/2), width, height, 
                                      linewidth=2, edgecolor=color, facecolor=color, alpha=0.3)
        ax.add_patch(output_rect)
        
        # Label for output - moved closer
        ax.text(output_x + width/2, pos_y - height/2 - 0.3, 
               f'Output Y{dilation_rate}', ha='center', fontsize=9, fontweight='bold')
        
        # Mathematical notation for output - moved closer
        ax.text(output_x + width/2, pos_y + height/2 + 0.3, 
               f'$\\mathbf{{Y}}_{{{dilation_rate}}}$', ha='center', fontsize=9, color=math_color)
        
        # Arrow from input to filter - adjusted for new positions
        ax.arrow(3.1, middle_section_y, 0.3, pos_y - middle_section_y, 
                head_width=0.12, head_length=0.11, fc=arrow_color, ec=arrow_color, width=0.03, 
                alpha=0.7, shape='right', length_includes_head=True)
        
        # Modified arrow from filter to output
        ax.arrow(pos_x + width + 0.05, pos_y, output_x - pos_x - width - 0.2, 0, 
                head_width=0.10, head_length=0.10, fc=arrow_color, ec=arrow_color, width=0.03, alpha=0.7)
    
    # Draw concatenation block with reduced height
    concat_x = 7.5
    concat_height = 2.0  # Reduced from 2.5
    concat_rect = patches.Rectangle((concat_x, middle_section_y - concat_height/2), 1, concat_height, 
                                   linewidth=2, edgecolor='#3498db', facecolor='#3498db', alpha=0.3)
    ax.add_patch(concat_rect)
    ax.text(concat_x + 0.5, middle_section_y - concat_height/2 - 0.3, 'Concatenate', 
           ha='center', fontsize=9, fontweight='bold')
    
    # Draw arrows from outputs to concatenation - improved curve for clearer display
    for _, pos_y, color, _ in filter_positions:
        # Calculate better curve for arrows to concatenation point
        ax.arrow(6.6, pos_y, concat_x - 6.7, middle_section_y - pos_y, 
                head_width=0.13, head_length=0.12, fc=arrow_color, ec=arrow_color, width=0.04, 
                alpha=0.7, shape='right', length_includes_head=True)
    
    # Draw 1×1 convolution block with reduced height
    conv_x = 9.0
    conv_rect = patches.Rectangle((conv_x, middle_section_y - concat_height/2), 1, concat_height, 
                                linewidth=2, edgecolor='#e74c3c', facecolor='#e74c3c', alpha=0.3)
    ax.add_patch(conv_rect)
    ax.text(conv_x + 0.5, middle_section_y - concat_height/2 - 0.3, '1×1 Conv', 
           ha='center', fontsize=9, fontweight='bold')
    ax.text(conv_x + 0.5, middle_section_y + concat_height/2 + 0.3, r'$C_{out}^{1 \times 1}$', 
           ha='center', fontsize=9, color=math_color, fontweight='bold')
    
    # Arrow from concat to conv
    ax.arrow(8.6, middle_section_y, 0.3, 0, head_width=0.13, head_length=0.12, 
            fc=arrow_color, ec=arrow_color, width=0.04, alpha=0.7)
    
    # Draw final output feature map with reduced height
    output_x = 10.5
    output_rect = patches.Rectangle((output_x, middle_section_y - concat_height/2), 1, concat_height, 
                                   linewidth=2, edgecolor=output_color, facecolor=output_color, alpha=0.3)
    ax.add_patch(output_rect)
    ax.text(output_x + 0.5, middle_section_y - concat_height/2 - 0.3, 'Final Output Y', 
           ha='center', fontsize=11, fontweight='bold')
    ax.text(output_x + 0.5, middle_section_y + concat_height/2 + 0.3, r'$\mathbf{Y}$', 
           ha='center', fontsize=9, color=math_color, fontweight='bold')
    
    # Arrow from conv to final output
    ax.arrow(10.1, middle_section_y, 0.3, 0, head_width=0.13, head_length=0.12, 
            fc=arrow_color, ec=arrow_color, width=0.04, alpha=0.7)

    # Add explanation of spatial varying filters - moved to conserve space
    note_rect = patches.Rectangle((8.5, 5.5), 3.5, 0.4, linewidth=1,
                                 edgecolor=highlight_color, facecolor='white', alpha=0.9)
    ax.add_patch(note_rect)
    ax.text(10.25, 5.7, "Spatial-Varying: Position-dependent filter weights",
           ha='center', va='center', fontsize=9, color=highlight_color, fontweight='bold')

    # Add simplified legend - tightened spacing
    legend_elements = [
        patches.Patch(facecolor=feature_color, alpha=0.3, edgecolor=feature_color, 
                     label='Input Features'),
        patches.Patch(facecolor=dilation_colors[0], alpha=0.3, edgecolor=dilation_colors[0], 
                     label='R=1 Filter'),
        patches.Patch(facecolor=dilation_colors[1], alpha=0.3, edgecolor=dilation_colors[1], 
                     label='R=2 Filter'),
        patches.Patch(facecolor=dilation_colors[2], alpha=0.3, edgecolor=dilation_colors[2], 
                     label='R=3 Filter'),
        patches.Patch(facecolor=output_color, alpha=0.3, edgecolor=output_color, 
                     label='Final Output')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02),
              fancybox=True, shadow=True, ncol=5, fontsize=8)
    
    # Set plot limits - adjusted for more compact layout
    ax.set_xlim(0, 12)
    ax.set_ylim(0.2, 6.0)  # Further reduced upper limit from 7.5 to 6.0
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])  
    
    plt.savefig('multi_scale_dilated_dconv.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    draw_dynamic_conv_diagram()
