"""
TinyNet Visual Architecture Diagram Generator

Creates beautiful, interactive HTML visualizations of the TinyNet architecture
with layer-by-layer details, parameter counts, and feature map dimensions.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Tuple
import json

from model import create_student_model, DepthwiseSeparableConv, ConvBlock
from config import Config


class ArchitectureVisualizer:
    """Generate visual diagrams of TinyNet architecture"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.layers = []
        self.total_params = 0
        
    def analyze_architecture(self, input_size: Tuple[int, int, int, int] = (1, 3, 224, 224)):
        """Analyze model architecture layer by layer"""
        current_shape = list(input_size)
        layer_id = 0
        
        # Input layer
        self.layers.append({
            'id': layer_id,
            'name': 'Input',
            'type': 'Input',
            'shape': f'{current_shape[1]}×{current_shape[2]}×{current_shape[3]}',
            'channels': current_shape[1],
            'spatial': f'{current_shape[2]}×{current_shape[3]}',
            'params': 0,
            'stage': 'Input',
            'color': '#e8f5e9'
        })
        layer_id += 1
        
        # Stem
        for i, layer in enumerate(self.model.stem):
            if isinstance(layer, nn.Conv2d):
                h_out = (current_shape[2] + 2*layer.padding[0] - layer.kernel_size[0]) // layer.stride[0] + 1
                w_out = (current_shape[3] + 2*layer.padding[1] - layer.kernel_size[1]) // layer.stride[1] + 1
                current_shape = [current_shape[0], layer.out_channels, h_out, w_out]
                
                params = sum(p.numel() for p in layer.parameters())
                self.total_params += params
                
                self.layers.append({
                    'id': layer_id,
                    'name': f'Stem Conv',
                    'type': 'Conv2d',
                    'shape': f'{current_shape[1]}×{current_shape[2]}×{current_shape[3]}',
                    'channels': current_shape[1],
                    'spatial': f'{current_shape[2]}×{current_shape[3]}',
                    'params': params,
                    'details': f'3×3, stride={layer.stride[0]}',
                    'stage': 'Stem',
                    'color': '#fff3e0'
                })
                layer_id += 1
                
            elif isinstance(layer, nn.GroupNorm):
                params = sum(p.numel() for p in layer.parameters())
                self.total_params += params
                
                self.layers.append({
                    'id': layer_id,
                    'name': f'Stem Norm',
                    'type': 'GroupNorm',
                    'shape': f'{current_shape[1]}×{current_shape[2]}×{current_shape[3]}',
                    'channels': current_shape[1],
                    'spatial': f'{current_shape[2]}×{current_shape[3]}',
                    'params': params,
                    'details': f'{layer.num_groups} groups',
                    'stage': 'Stem',
                    'color': '#fff3e0'
                })
                layer_id += 1
                
            elif isinstance(layer, nn.ReLU):
                self.layers.append({
                    'id': layer_id,
                    'name': f'Stem ReLU',
                    'type': 'ReLU',
                    'shape': f'{current_shape[1]}×{current_shape[2]}×{current_shape[3]}',
                    'channels': current_shape[1],
                    'spatial': f'{current_shape[2]}×{current_shape[3]}',
                    'params': 0,
                    'details': 'activation',
                    'stage': 'Stem',
                    'color': '#fff3e0'
                })
                layer_id += 1
        
        # Stage colors
        stage_colors = ['#e3f2fd', '#f3e5f5', '#ffe0b2', '#ffcdd2']
        
        # Stages
        for stage_idx, stage in enumerate(self.model.stages):
            stage_name = f'Stage {stage_idx + 1}'
            stage_color = stage_colors[stage_idx % len(stage_colors)]
            
            for block_idx, block in enumerate(stage):
                block_name = f'Block {block_idx + 1}'
                
                # Convolution
                if hasattr(block, 'conv'):
                    conv = block.conv
                    
                    if isinstance(conv, DepthwiseSeparableConv):
                        # Depthwise
                        dw = conv.depthwise
                        h_out = (current_shape[2] + 2*dw.padding[0] - dw.kernel_size[0]) // dw.stride[0] + 1
                        w_out = (current_shape[3] + 2*dw.padding[1] - dw.kernel_size[1]) // dw.stride[1] + 1
                        current_shape = [current_shape[0], dw.out_channels, h_out, w_out]
                        
                        params = sum(p.numel() for p in dw.parameters())
                        self.total_params += params
                        
                        self.layers.append({
                            'id': layer_id,
                            'name': f'{stage_name} {block_name} DW',
                            'type': 'DepthwiseConv',
                            'shape': f'{current_shape[1]}×{current_shape[2]}×{current_shape[3]}',
                            'channels': current_shape[1],
                            'spatial': f'{current_shape[2]}×{current_shape[3]}',
                            'params': params,
                            'details': f'3×3, stride={dw.stride[0]}, groups={dw.groups}',
                            'stage': stage_name,
                            'color': stage_color
                        })
                        layer_id += 1
                        
                        # Pointwise
                        pw = conv.pointwise
                        current_shape = [current_shape[0], pw.out_channels, current_shape[2], current_shape[3]]
                        
                        params = sum(p.numel() for p in pw.parameters())
                        self.total_params += params
                        
                        self.layers.append({
                            'id': layer_id,
                            'name': f'{stage_name} {block_name} PW',
                            'type': 'PointwiseConv',
                            'shape': f'{current_shape[1]}×{current_shape[2]}×{current_shape[3]}',
                            'channels': current_shape[1],
                            'spatial': f'{current_shape[2]}×{current_shape[3]}',
                            'params': params,
                            'details': f'1×1, {current_shape[1]} channels',
                            'stage': stage_name,
                            'color': stage_color
                        })
                        layer_id += 1
                        
                    elif isinstance(conv, nn.Conv2d):
                        h_out = (current_shape[2] + 2*conv.padding[0] - conv.kernel_size[0]) // conv.stride[0] + 1
                        w_out = (current_shape[3] + 2*conv.padding[1] - conv.kernel_size[1]) // conv.stride[1] + 1
                        current_shape = [current_shape[0], conv.out_channels, h_out, w_out]
                        
                        params = sum(p.numel() for p in conv.parameters())
                        self.total_params += params
                        
                        self.layers.append({
                            'id': layer_id,
                            'name': f'{stage_name} {block_name} Conv',
                            'type': 'Conv2d',
                            'shape': f'{current_shape[1]}×{current_shape[2]}×{current_shape[3]}',
                            'channels': current_shape[1],
                            'spatial': f'{current_shape[2]}×{current_shape[3]}',
                            'params': params,
                            'details': f'3×3, stride={conv.stride[0]}',
                            'stage': stage_name,
                            'color': stage_color
                        })
                        layer_id += 1
                
                # Normalization
                if hasattr(block, 'norm') and not isinstance(block.norm, nn.Identity):
                    norm = block.norm
                    params = sum(p.numel() for p in norm.parameters())
                    self.total_params += params
                    
                    norm_type = 'GroupNorm' if isinstance(norm, nn.GroupNorm) else 'BatchNorm'
                    details = f'{norm.num_groups} groups' if isinstance(norm, nn.GroupNorm) else 'batch norm'
                    
                    self.layers.append({
                        'id': layer_id,
                        'name': f'{stage_name} {block_name} Norm',
                        'type': norm_type,
                        'shape': f'{current_shape[1]}×{current_shape[2]}×{current_shape[3]}',
                        'channels': current_shape[1],
                        'spatial': f'{current_shape[2]}×{current_shape[3]}',
                        'params': params,
                        'details': details,
                        'stage': stage_name,
                        'color': stage_color
                    })
                    layer_id += 1
                
                # Activation
                if hasattr(block, 'act') and not isinstance(block.act, nn.Identity):
                    act = block.act
                    self.layers.append({
                        'id': layer_id,
                        'name': f'{stage_name} {block_name} Act',
                        'type': act.__class__.__name__,
                        'shape': f'{current_shape[1]}×{current_shape[2]}×{current_shape[3]}',
                        'channels': current_shape[1],
                        'spatial': f'{current_shape[2]}×{current_shape[3]}',
                        'params': 0,
                        'details': 'activation',
                        'stage': stage_name,
                        'color': stage_color
                    })
                    layer_id += 1
        
        # Global Average Pooling
        current_shape = [current_shape[0], current_shape[1], 1, 1]
        self.layers.append({
            'id': layer_id,
            'name': 'Global Avg Pool',
            'type': 'AdaptiveAvgPool2d',
            'shape': f'{current_shape[1]}×{current_shape[2]}×{current_shape[3]}',
            'channels': current_shape[1],
            'spatial': f'{current_shape[2]}×{current_shape[3]}',
            'params': 0,
            'details': 'adaptive pooling',
            'stage': 'Head',
            'color': '#f1f8e9'
        })
        layer_id += 1
        
        # Classifier
        classifier = self.model.classifier
        params = sum(p.numel() for p in classifier.parameters())
        self.total_params += params
        
        current_shape = [current_shape[0], classifier.out_channels, 1, 1]
        self.layers.append({
            'id': layer_id,
            'name': 'Classifier',
            'type': 'Conv2d 1×1',
            'shape': f'{current_shape[1]}×{current_shape[2]}×{current_shape[3]}',
            'channels': current_shape[1],
            'spatial': f'{current_shape[2]}×{current_shape[3]}',
            'params': params,
            'details': f'{classifier.out_channels} classes',
            'stage': 'Head',
            'color': '#f1f8e9'
        })
        layer_id += 1
        
        # Output
        self.layers.append({
            'id': layer_id,
            'name': 'Output',
            'type': 'Flatten',
            'shape': f'{current_shape[1]}',
            'channels': current_shape[1],
            'spatial': 'scalar',
            'params': 0,
            'details': f'{current_shape[1]} logits',
            'stage': 'Output',
            'color': '#c8e6c9'
        })
    
    def generate_html(self, save_path: str = 'tinynet_architecture.html'):
        """Generate interactive HTML visualization"""
        
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TinyNet Architecture Visualization</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }
        
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        }
        
        .stat-label {
            color: #666;
            font-size: 0.9em;
            margin-bottom: 5px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .stat-value {
            color: #333;
            font-size: 2em;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .architecture {
            padding: 30px;
        }
        
        .stage-section {
            margin-bottom: 40px;
        }
        
        .stage-title {
            font-size: 1.5em;
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .stage-icon {
            width: 30px;
            height: 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        
        .layers-grid {
            display: grid;
            gap: 15px;
        }
        
        .layer-card {
            background: white;
            border-left: 4px solid #667eea;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: all 0.3s;
            cursor: pointer;
        }
        
        .layer-card:hover {
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
            transform: translateX(5px);
        }
        
        .layer-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .layer-name {
            font-weight: bold;
            font-size: 1.1em;
            color: #333;
        }
        
        .layer-type {
            background: #667eea;
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 500;
        }
        
        .layer-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        
        .detail-item {
            display: flex;
            flex-direction: column;
        }
        
        .detail-label {
            color: #666;
            font-size: 0.85em;
            margin-bottom: 2px;
        }
        
        .detail-value {
            color: #333;
            font-weight: 600;
        }
        
        .params-badge {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 2px 8px;
            border-radius: 8px;
            font-size: 0.85em;
            font-weight: bold;
        }
        
        .arrow {
            text-align: center;
            color: #667eea;
            font-size: 2em;
            margin: 10px 0;
        }
        
        .legend {
            padding: 30px;
            background: #f8f9fa;
            border-top: 1px solid #dee2e6;
        }
        
        .legend-title {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }
        
        .legend-items {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .legend-color {
            width: 20px;
            height: 20px;
            border-radius: 4px;
            border: 1px solid #ddd;
        }
        
        @media (max-width: 768px) {
            .layer-details {
                grid-template-columns: 1fr;
            }
            
            .stats {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 TinyNet Architecture</h1>
            <p>Lightweight CNN for Industrial Defect Detection</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Total Layers</div>
                <div class="stat-value">{{TOTAL_LAYERS}}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Parameters</div>
                <div class="stat-value">{{TOTAL_PARAMS}}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Model Size</div>
                <div class="stat-value">{{MODEL_SIZE}}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Input Size</div>
                <div class="stat-value">224×224</div>
            </div>
        </div>
        
        <div class="architecture">
            {{LAYERS_HTML}}
        </div>
        
        <div class="legend">
            <div class="legend-title">Layer Type Legend</div>
            <div class="legend-items">
                <div class="legend-item">
                    <div class="legend-color" style="background: #e8f5e9;"></div>
                    <span>Input</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #fff3e0;"></div>
                    <span>Stem</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #e3f2fd;"></div>
                    <span>Stage 1</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #f3e5f5;"></div>
                    <span>Stage 2</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #ffe0b2;"></div>
                    <span>Stage 3</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #ffcdd2;"></div>
                    <span>Stage 4</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #f1f8e9;"></div>
                    <span>Head</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #c8e6c9;"></div>
                    <span>Output</span>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
        
        # Group layers by stage
        stages = {}
        for layer in self.layers:
            stage = layer['stage']
            if stage not in stages:
                stages[stage] = []
            stages[stage].append(layer)
        
        # Generate layers HTML
        layers_html = ""
        for stage_name, stage_layers in stages.items():
            layers_html += f'<div class="stage-section">\n'
            layers_html += f'    <div class="stage-title">\n'
            layers_html += f'        <div class="stage-icon">{len(stage_layers)}</div>\n'
            layers_html += f'        <span>{stage_name}</span>\n'
            layers_html += f'    </div>\n'
            layers_html += f'    <div class="layers-grid">\n'
            
            for i, layer in enumerate(stage_layers):
                layers_html += f'        <div class="layer-card" style="background-color: {layer["color"]};">\n'
                layers_html += f'            <div class="layer-header">\n'
                layers_html += f'                <div class="layer-name">{layer["name"]}</div>\n'
                layers_html += f'                <div class="layer-type">{layer["type"]}</div>\n'
                layers_html += f'            </div>\n'
                layers_html += f'            <div class="layer-details">\n'
                layers_html += f'                <div class="detail-item">\n'
                layers_html += f'                    <div class="detail-label">Output Shape</div>\n'
                layers_html += f'                    <div class="detail-value">{layer["shape"]}</div>\n'
                layers_html += f'                </div>\n'
                layers_html += f'                <div class="detail-item">\n'
                layers_html += f'                    <div class="detail-label">Channels</div>\n'
                layers_html += f'                    <div class="detail-value">{layer["channels"]}</div>\n'
                layers_html += f'                </div>\n'
                layers_html += f'                <div class="detail-item">\n'
                layers_html += f'                    <div class="detail-label">Spatial Size</div>\n'
                layers_html += f'                    <div class="detail-value">{layer["spatial"]}</div>\n'
                layers_html += f'                </div>\n'
                if layer["params"] > 0:
                    layers_html += f'                <div class="detail-item">\n'
                    layers_html += f'                    <div class="detail-label">Parameters</div>\n'
                    layers_html += f'                    <div class="detail-value"><span class="params-badge">{layer["params"]:,}</span></div>\n'
                    layers_html += f'                </div>\n'
                if 'details' in layer:
                    layers_html += f'                <div class="detail-item">\n'
                    layers_html += f'                    <div class="detail-label">Details</div>\n'
                    layers_html += f'                    <div class="detail-value">{layer["details"]}</div>\n'
                    layers_html += f'                </div>\n'
                layers_html += f'            </div>\n'
                layers_html += f'        </div>\n'
                
                # Add arrow between layers (except last in stage)
                if i < len(stage_layers) - 1:
                    layers_html += f'        <div class="arrow">↓</div>\n'
            
            layers_html += f'    </div>\n'
            layers_html += f'</div>\n'
        
        # Calculate model size
        model_size_mb = self.total_params * 4 / (1024 ** 2)  # FP32
        
        # Replace placeholders
        html = html_template.replace('{{TOTAL_LAYERS}}', str(len(self.layers)))
        html = html.replace('{{TOTAL_PARAMS}}', f'{self.total_params:,}')
        html = html.replace('{{MODEL_SIZE}}', f'{model_size_mb:.2f} MB')
        html = html.replace('{{LAYERS_HTML}}', layers_html)
        
        # Save HTML
        with open(save_path, 'w') as f:
            f.write(html)
        
        print(f"✓ HTML visualization saved to {save_path}")
        print(f"  Total layers: {len(self.layers)}")
        print(f"  Total parameters: {self.total_params:,}")
        print(f"  Model size (FP32): {model_size_mb:.2f} MB")
        
        return save_path


def create_architecture_diagram(
    model: nn.Module = None,
    save_path: str = 'tinynet_architecture.html',
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224)
) -> str:
    """
    Create visual architecture diagram
    
    Args:
        model: TinyNet model (creates new if None)
        save_path: Path to save HTML file
        input_size: Input tensor size
    
    Returns:
        Path to saved HTML file
    """
    if model is None:
        print("Creating TinyNet model...")
        model = create_student_model(num_classes=6)
    
    print("\nAnalyzing architecture...")
    visualizer = ArchitectureVisualizer(model)
    visualizer.analyze_architecture(input_size)
    
    print("\nGenerating HTML visualization...")
    output_path = visualizer.generate_html(save_path)
    
    return output_path


if __name__ == "__main__":
    print("="*80)
    print("TinyNet Architecture Visualizer")
    print("="*80)
    
    # Create visualization
    html_path = create_architecture_diagram(
        save_path='tinynet_architecture.html'
    )
    
    print(f"\n{'='*80}")
    print("✓ Visualization complete!")
    print(f"  Open {html_path} in your browser to view the interactive diagram")
    print(f"{'='*80}")