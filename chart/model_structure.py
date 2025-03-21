from graphviz import Digraph

# 创建有向图
dot = Digraph(comment='Xception CNN')
dot.attr(rankdir='LR', nodesep='0.15', ranksep='0.25')  # 修改为LR(左至右)
dot.attr('node', shape='box', style='rounded,filled', fontname='Helvetica', fontsize='9')

# 颜色定义 - 使用更浅的颜色
entry_color = '#D6E4F0'  # 更浅的蓝色
middle_color = '#E1D4F0'  # 更浅的紫色
exit_color = '#D9D6F0'    # 更浅的石板蓝
sda_color = '#E0D8FF'     # 更浅的亮紫色
shortcut_color = '#F0F0F0' # 更浅的灰色
dyn_conv_color = '#F0E6D4' # 新增：动态卷积颜色（浅橙色）

# 动态卷积子图
with dot.subgraph(name='cluster_dyconv') as dyconv:
    dyconv.attr(label='Dynamic Convolution', style='filled', color=dyn_conv_color, bgcolor='#FFFAF0')
    
    dyconv.node('DC_input', 'Input\nFeature', shape='ellipse')
    dyconv.node('DC_attn', 'Attention\nWeights', fillcolor=dyn_conv_color)
    dyconv.node('DC_k1', 'Kernel 1', fillcolor=dyn_conv_color)
    dyconv.node('DC_k2', 'Kernel 2', fillcolor=dyn_conv_color)
    dyconv.node('DC_k3', '...', fillcolor=dyn_conv_color)
    dyconv.node('DC_conv', 'Dynamic\nWeighting', fillcolor=dyn_conv_color)
    dyconv.node('DC_output', 'Output\nFeature', shape='ellipse')
    
    dyconv.edges([('DC_input', 'DC_conv'),
                  ('DC_input', 'DC_attn'),
                  ('DC_attn', 'DC_conv'),
                  ('DC_k1', 'DC_conv'),
                  ('DC_k2', 'DC_conv'),
                  ('DC_k3', 'DC_conv'),
                  ('DC_conv', 'DC_output')])

# Entry Flow 子图
with dot.subgraph(name='cluster_entry') as entry:
    entry.attr(label='Entry Flow', style='filled', color=entry_color, bgcolor='#F8FCFF')
    
    entry.node('input', 'Input', shape='ellipse')
    entry.node('E1', 'Conv 128\n1x1', fillcolor=entry_color)
    entry.edge('input', 'E1')
    
    # 简化为只显示一个Entry模块
    entry.node('E_1', 'Dconv\n3x3', fillcolor=sda_color)
    entry.node('E_2', 'Dconv\n3x3', fillcolor=sda_color)
    entry.node('E_3', 'Conv 728\n1x1', fillcolor=entry_color)
    entry.node('E_s', 'Shortcut', fillcolor=shortcut_color)
    
    entry.edges([('E1', 'E_1'),
                 ('E_1', 'E_2'),
                 ('E_2', 'E_3'),
                 ('E1', 'E_s'),
                 ('E_s', 'E_3')])
    
    entry.node('E_more', '...2 more\nsimilar blocks', shape='note', fillcolor='#FFFAFA')

# Middle Flow 子图
with dot.subgraph(name='cluster_middle') as middle:
    middle.attr(label='Middle Flow (×16)', style='filled', color=middle_color, bgcolor='#F6F0FF')
    middle.node('M1', 'Dconv\n3x3', fillcolor=sda_color)
    middle.node('M3', 'Dconv\n3x3', fillcolor=sda_color)
    middle.node('Ms', 'Shortcut', fillcolor=shortcut_color)
    middle.edges([('M1', 'M3'), ('M1', 'Ms'), ('Ms', 'M3')])
    
    # 连接Entry和Middle
    dot.edge('E_3', 'M1')
    dot.edge('E_more', 'E_3', style='dashed', color='gray')

# Exit Flow 子图
with dot.subgraph(name='cluster_exit') as exit:
    exit.attr(label='Exit Flow', style='filled', color=exit_color, bgcolor='#F8F8FF')
    
    exit.node('X1', 'Conv\n1024', fillcolor=exit_color)
    exit.node('X3', 'Dconv\n2048', fillcolor=sda_color)
    exit.node('X4', 'GAP', shape='component')
    exit.node('X5', 'Output', shape='ellipse')
    
    exit.edges([('X1', 'X3'), ('X3', 'X4'), ('X4', 'X5')])
    
    # 连接Middle和Exit
    dot.edge('M3', 'X1')

# 添加说明
dot.node('note_dc', 'Dynamic Conv: Adaptive kernel selection\nbased on input features', shape='note', fillcolor='#FFFAFA')
dot.edge('note_dc', 'DC_conv', style='dashed', arrowhead='none')

# 保存并渲染
dot.render('architecture', format='png', cleanup=True)
print("流程图已生成：architecture.png")