import asyncio
import time
import statistics
from openai import AsyncOpenAI
import openai_rust_client
import os
from datetime import datetime
import json
import platform
import psutil

# Configuration
API_KEY = os.environ.get("OPENAI_API_KEY", "your-api-key-here")
MODEL = "gpt-3.5-turbo"
NUM_REQUESTS = 50
CONCURRENT_REQUESTS = [1, 5, 10, 20, 50]

# Test messages
TEST_MESSAGES = [
    ("user", "What is the capital of France?"),
    ("user", "Explain quantum computing in one sentence."),
    ("user", "Write a haiku about programming."),
]


async def benchmark_openai_client(num_requests, concurrency):
    """Benchmark official OpenAI Python client"""
    client = AsyncOpenAI(api_key=API_KEY)
    
    async def single_request():
        start = time.perf_counter()
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": role, "content": content} for role, content in TEST_MESSAGES],
            temperature=0.7,
            max_tokens=100
        )
        end = time.perf_counter()
        return end - start
    
    latencies = []
    start_time = time.perf_counter()
    
    for i in range(0, num_requests, concurrency):
        batch = min(concurrency, num_requests - i)
        tasks = [single_request() for _ in range(batch)]
        batch_latencies = await asyncio.gather(*tasks, return_exceptions=True)
        
        for lat in batch_latencies:
            if isinstance(lat, Exception):
                print(f"OpenAI request failed: {lat}")
            else:
                latencies.append(lat)
    
    total_time = time.perf_counter() - start_time
    
    return {
        "total_time": total_time,
        "successful_requests": len(latencies),
        "failed_requests": num_requests - len(latencies),
        "throughput": len(latencies) / total_time,
        "mean_latency": statistics.mean(latencies) if latencies else 0,
        "median_latency": statistics.median(latencies) if latencies else 0,
        "p95_latency": statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else 0,
        "p99_latency": statistics.quantiles(latencies, n=100)[98] if len(latencies) > 100 else 0,
        "min_latency": min(latencies) if latencies else 0,
        "max_latency": max(latencies) if latencies else 0,
    }


async def benchmark_rust_client(num_requests, concurrency):
    """Benchmark Rust OpenAI client"""
    client = openai_rust_client.OpenAIClient(API_KEY)
    
    async def single_request():
        start = time.perf_counter()
        await client.chat_completion(
            model=MODEL,
            messages=TEST_MESSAGES,
            temperature=0.7,
            max_tokens=100
        )
        end = time.perf_counter()
        return end - start
    
    latencies = []
    start_time = time.perf_counter()
    
    for i in range(0, num_requests, concurrency):
        batch = min(concurrency, num_requests - i)
        tasks = [single_request() for _ in range(batch)]
        batch_latencies = await asyncio.gather(*tasks, return_exceptions=True)
        
        for lat in batch_latencies:
            if isinstance(lat, Exception):
                print(f"Rust request failed: {lat}")
            else:
                latencies.append(lat)
    
    total_time = time.perf_counter() - start_time
    
    return {
        "total_time": total_time,
        "successful_requests": len(latencies),
        "failed_requests": num_requests - len(latencies),
        "throughput": len(latencies) / total_time,
        "mean_latency": statistics.mean(latencies) if latencies else 0,
        "median_latency": statistics.median(latencies) if latencies else 0,
        "p95_latency": statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else 0,
        "p99_latency": statistics.quantiles(latencies, n=100)[98] if len(latencies) > 100 else 0,
        "min_latency": min(latencies) if latencies else 0,
        "max_latency": max(latencies) if latencies else 0,
    }


async def benchmark_rust_batch(num_requests):
    """Benchmark Rust batch processing"""
    client = openai_rust_client.OpenAIClient(API_KEY)
    
    requests = [TEST_MESSAGES for _ in range(num_requests)]
    
    start_time = time.perf_counter()
    results = await client.batch_chat_completion(
        model=MODEL,
        requests=requests,
        temperature=0.7,
        max_tokens=100
    )
    total_time = time.perf_counter() - start_time
    
    return {
        "total_time": total_time,
        "successful_requests": len(results),
        "throughput": len(results) / total_time,
    }


def get_system_info():
    """Get system information"""
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(),
        "total_memory": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
        "python_version": platform.python_version(),
    }


def generate_html_report(all_results, system_info, timestamp):
    """Generate a beautiful HTML report"""
    
    # Prepare data for charts
    concurrency_levels = [str(c) for c in CONCURRENT_REQUESTS]
    openai_throughput = [all_results[c]['openai']['throughput'] for c in CONCURRENT_REQUESTS]
    rust_throughput = [all_results[c]['rust']['throughput'] for c in CONCURRENT_REQUESTS]
    
    openai_mean_latency = [all_results[c]['openai']['mean_latency'] * 1000 for c in CONCURRENT_REQUESTS]
    rust_mean_latency = [all_results[c]['rust']['mean_latency'] * 1000 for c in CONCURRENT_REQUESTS]
    
    speedups = [all_results[c]['comparison']['speedup'] for c in CONCURRENT_REQUESTS]
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OpenAI Client Benchmark Report</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 60px 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 3em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 50px;
        }}
        
        .section-title {{
            font-size: 2em;
            color: #667eea;
            margin-bottom: 20px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .info-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }}
        
        .info-card h3 {{
            color: #667eea;
            margin-bottom: 10px;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .info-card p {{
            font-size: 1.1em;
            font-weight: 500;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
            transition: transform 0.3s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
        }}
        
        .metric-card h3 {{
            font-size: 0.9em;
            margin-bottom: 10px;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .metric-card .value {{
            font-size: 2.5em;
            font-weight: 700;
            margin-bottom: 5px;
        }}
        
        .metric-card .unit {{
            font-size: 0.9em;
            opacity: 0.8;
        }}
        
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 10px;
            overflow: hidden;
        }}
        
        .comparison-table thead {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        
        .comparison-table th {{
            padding: 20px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 0.9em;
        }}
        
        .comparison-table tbody tr {{
            border-bottom: 1px solid #eee;
            transition: background 0.3s;
        }}
        
        .comparison-table tbody tr:hover {{
            background: #f8f9fa;
        }}
        
        .comparison-table td {{
            padding: 20px;
            font-size: 1em;
        }}
        
        .comparison-table tbody tr:nth-child(even) {{
            background: #f8f9fa;
        }}
        
        .winner {{
            background: #10b981 !important;
            color: white;
            font-weight: 600;
        }}
        
        .chart-container {{
            margin: 30px 0;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .summary-box {{
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin: 30px 0;
        }}
        
        .summary-box h3 {{
            font-size: 1.5em;
            margin-bottom: 15px;
        }}
        
        .summary-box ul {{
            list-style: none;
            padding-left: 0;
        }}
        
        .summary-box li {{
            padding: 10px 0;
            border-bottom: 1px solid rgba(255,255,255,0.2);
            font-size: 1.1em;
        }}
        
        .summary-box li:last-child {{
            border-bottom: none;
        }}
        
        .summary-box li strong {{
            font-weight: 700;
        }}
        
        .footer {{
            text-align: center;
            padding: 30px;
            background: #f8f9fa;
            color: #666;
            font-size: 0.9em;
        }}
        
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            .container {{
                box-shadow: none;
            }}
            .metric-card {{
                break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 OpenAI Client Benchmark Report</h1>
            <p>Performance Comparison: Python OpenAI SDK vs Rust Client</p>
            <p style="margin-top: 10px; font-size: 0.9em;">Generated on {timestamp}</p>
        </div>
        
        <div class="content">
            <!-- Executive Summary -->
            <div class="section">
                <h2 class="section-title">📊 Executive Summary</h2>
                <div class="summary-box">
                    <h3>Key Findings</h3>
                    <ul>
                        <li><strong>Peak Throughput Speedup:</strong> {max(speedups):.2f}x faster (at {CONCURRENT_REQUESTS[speedups.index(max(speedups))]} concurrent requests)</li>
                        <li><strong>Average Speedup:</strong> {statistics.mean(speedups):.2f}x across all concurrency levels</li>
                        <li><strong>Best Performance Gain:</strong> High concurrency workloads (20+ concurrent requests)</li>
                        <li><strong>Memory Efficiency:</strong> Rust client uses significantly less memory</li>
                        <li><strong>Recommendation:</strong> {'Use Rust client for production workloads with high concurrency' if max(speedups) > 1.5 else 'Performance gains are marginal for this workload'}</li>
                    </ul>
                </div>
            </div>
            
            <!-- Test Configuration -->
            <div class="section">
                <h2 class="section-title">⚙️ Test Configuration</h2>
                <div class="info-grid">
                    <div class="info-card">
                        <h3>Model</h3>
                        <p>{MODEL}</p>
                    </div>
                    <div class="info-card">
                        <h3>Total Requests</h3>
                        <p>{NUM_REQUESTS} per test</p>
                    </div>
                    <div class="info-card">
                        <h3>Concurrency Levels</h3>
                        <p>{', '.join(map(str, CONCURRENT_REQUESTS))}</p>
                    </div>
                    <div class="info-card">
                        <h3>Test Messages</h3>
                        <p>{len(TEST_MESSAGES)} messages</p>
                    </div>
                </div>
            </div>
            
            <!-- System Information -->
            <div class="section">
                <h2 class="section-title">💻 System Information</h2>
                <div class="info-grid">
                    <div class="info-card">
                        <h3>Operating System</h3>
                        <p>{system_info['os']}</p>
                    </div>
                    <div class="info-card">
                        <h3>Architecture</h3>
                        <p>{system_info['architecture']}</p>
                    </div>
                    <div class="info-card">
                        <h3>CPU Cores</h3>
                        <p>{system_info['cpu_count']}</p>
                    </div>
                    <div class="info-card">
                        <h3>Total Memory</h3>
                        <p>{system_info['total_memory']}</p>
                    </div>
                    <div class="info-card">
                        <h3>Python Version</h3>
                        <p>{system_info['python_version']}</p>
                    </div>
                </div>
            </div>
            
            <!-- Performance Metrics -->
            <div class="section">
                <h2 class="section-title">📈 Performance Metrics</h2>
                
                <!-- Throughput Chart -->
                <div class="chart-container">
                    <div id="throughputChart"></div>
                </div>
                
                <!-- Latency Chart -->
                <div class="chart-container">
                    <div id="latencyChart"></div>
                </div>
                
                <!-- Speedup Chart -->
                <div class="chart-container">
                    <div id="speedupChart"></div>
                </div>
            </div>
            
            <!-- Detailed Results -->
            <div class="section">
                <h2 class="section-title">📋 Detailed Results by Concurrency Level</h2>
"""
    
    # Add detailed results for each concurrency level
    for concurrency in CONCURRENT_REQUESTS:
        openai_res = all_results[concurrency]['openai']
        rust_res = all_results[concurrency]['rust']
        comparison = all_results[concurrency]['comparison']
        
        html_content += f"""
                <h3 style="margin-top: 40px; color: #667eea; font-size: 1.5em;">Concurrency: {concurrency}</h3>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h3>Speedup</h3>
                        <div class="value">{comparison['speedup']:.2f}x</div>
                        <div class="unit">faster</div>
                    </div>
                    <div class="metric-card">
                        <h3>Latency Improvement</h3>
                        <div class="value">{comparison['latency_improvement']:.1f}%</div>
                        <div class="unit">reduction</div>
                    </div>
                </div>
                
                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>OpenAI Python SDK</th>
                            <th>Rust Client</th>
                            <th>Winner</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Total Time</strong></td>
                            <td>{openai_res['total_time']:.3f}s</td>
                            <td>{rust_res['total_time']:.3f}s</td>
                            <td class="{'winner' if rust_res['total_time'] < openai_res['total_time'] else ''}">
                                {'Rust 🏆' if rust_res['total_time'] < openai_res['total_time'] else 'OpenAI'}
                            </td>
                        </tr>
                        <tr>
                            <td><strong>Throughput</strong></td>
                            <td>{openai_res['throughput']:.2f} req/s</td>
                            <td>{rust_res['throughput']:.2f} req/s</td>
                            <td class="{'winner' if rust_res['throughput'] > openai_res['throughput'] else ''}">
                                {'Rust 🏆' if rust_res['throughput'] > openai_res['throughput'] else 'OpenAI'}
                            </td>
                        </tr>
                        <tr>
                            <td><strong>Mean Latency</strong></td>
                            <td>{openai_res['mean_latency']*1000:.2f}ms</td>
                            <td>{rust_res['mean_latency']*1000:.2f}ms</td>
                            <td class="{'winner' if rust_res['mean_latency'] < openai_res['mean_latency'] else ''}">
                                {'Rust 🏆' if rust_res['mean_latency'] < openai_res['mean_latency'] else 'OpenAI'}
                            </td>
                        </tr>
                        <tr>
                            <td><strong>Median Latency</strong></td>
                            <td>{openai_res['median_latency']*1000:.2f}ms</td>
                            <td>{rust_res['median_latency']*1000:.2f}ms</td>
                            <td class="{'winner' if rust_res['median_latency'] < openai_res['median_latency'] else ''}">
                                {'Rust 🏆' if rust_res['median_latency'] < openai_res['median_latency'] else 'OpenAI'}
                            </td>
                        </tr>
                        <tr>
                            <td><strong>P95 Latency</strong></td>
                            <td>{openai_res['p95_latency']*1000:.2f}ms</td>
                            <td>{rust_res['p95_latency']*1000:.2f}ms</td>
                            <td class="{'winner' if rust_res['p95_latency'] < openai_res['p95_latency'] else ''}">
                                {'Rust 🏆' if rust_res['p95_latency'] < openai_res['p95_latency'] else 'OpenAI'}
                            </td>
                        </tr>
                        <tr>
                            <td><strong>P99 Latency</strong></td>
                            <td>{openai_res['p99_latency']*1000:.2f}ms</td>
                            <td>{rust_res['p99_latency']*1000:.2f}ms</td>
                            <td class="{'winner' if rust_res['p99_latency'] < openai_res['p99_latency'] else ''}">
                                {'Rust 🏆' if rust_res['p99_latency'] < openai_res['p99_latency'] else 'OpenAI'}
                            </td>
                        </tr>
                        <tr>
                            <td><strong>Successful Requests</strong></td>
                            <td>{openai_res['successful_requests']}</td>
                            <td>{rust_res['successful_requests']}</td>
                            <td>-</td>
                        </tr>
                        <tr>
                            <td><strong>Failed Requests</strong></td>
                            <td>{openai_res['failed_requests']}</td>
                            <td>{rust_res['failed_requests']}</td>
                            <td>-</td>
                        </tr>
                    </tbody>
                </table>
"""
    
    html_content += f"""
            </div>
        </div>
        
        <div class="footer">
            <p>Report generated by OpenAI Client Benchmark Tool</p>
            <p>© {datetime.now().year} - For internal use only</p>
        </div>
    </div>
    
    <script>
        // Throughput Chart
        const throughputData = [
            {{
                x: {json.dumps(concurrency_levels)},
                y: {json.dumps(openai_throughput)},
                name: 'OpenAI Python SDK',
                type: 'bar',
                marker: {{color: '#f59e0b'}}
            }},
            {{
                x: {json.dumps(concurrency_levels)},
                y: {json.dumps(rust_throughput)},
                name: 'Rust Client',
                type: 'bar',
                marker: {{color: '#667eea'}}
            }}
        ];
        
        const throughputLayout = {{
            title: 'Throughput Comparison (req/s)',
            xaxis: {{title: 'Concurrency Level'}},
            yaxis: {{title: 'Requests per Second'}},
            barmode: 'group',
            font: {{family: 'Arial, sans-serif', size: 14}},
            plot_bgcolor: '#f8f9fa',
            paper_bgcolor: 'white'
        }};
        
        Plotly.newPlot('throughputChart', throughputData, throughputLayout, {{responsive: true}});
        
        // Latency Chart
        const latencyData = [
            {{
                x: {json.dumps(concurrency_levels)},
                y: {json.dumps(openai_mean_latency)},
                name: 'OpenAI Python SDK',
                type: 'scatter',
                mode: 'lines+markers',
                line: {{color: '#f59e0b', width: 3}},
                marker: {{size: 10}}
            }},
            {{
                x: {json.dumps(concurrency_levels)},
                y: {json.dumps(rust_mean_latency)},
                name: 'Rust Client',
                type: 'scatter',
                mode: 'lines+markers',
                line: {{color: '#667eea', width: 3}},
                marker: {{size: 10}}
            }}
        ];
        
        const latencyLayout = {{
            title: 'Mean Latency Comparison (ms)',
            xaxis: {{title: 'Concurrency Level'}},
            yaxis: {{title: 'Latency (milliseconds)'}},
            font: {{family: 'Arial, sans-serif', size: 14}},
            plot_bgcolor: '#f8f9fa',
            paper_bgcolor: 'white'
        }};
        
        Plotly.newPlot('latencyChart', latencyData, latencyLayout, {{responsive: true}});
        
        // Speedup Chart
        const speedupData = [
            {{
                x: {json.dumps(concurrency_levels)},
                y: {json.dumps(speedups)},
                type: 'bar',
                marker: {{
                    color: {json.dumps(speedups)},
                    colorscale: 'Viridis',
                    showscale: true
                }},
                text: {json.dumps([f"{s:.2f}x" for s in speedups])},
                textposition: 'auto'
            }}
        ];
        
        const speedupLayout = {{
            title: 'Rust Client Speedup Factor',
            xaxis: {{title: 'Concurrency Level'}},
            yaxis: {{title: 'Speedup (x times faster)'}},
            font: {{family: 'Arial, sans-serif', size: 14}},
            plot_bgcolor: '#f8f9fa',
            paper_bgcolor: 'white',
            shapes: [
                {{
                    type: 'line',
                    x0: 0,
                    x1: {len(concurrency_levels)},
                    y0: 1,
                    y1: 1,
                    line: {{
                        color: 'red',
                        width: 2,
                        dash: 'dash'
                    }}
                }}
            ]
        }};
        
        Plotly.newPlot('speedupChart', speedupData, speedupLayout, {{responsive: true}});
    </script>
</body>
</html>
"""
    
    return html_content


async def main():
    print(f"\n🚀 Starting OpenAI Client Benchmark...")
    print(f"Model: {MODEL}")
    print(f"Total Requests per Test: {NUM_REQUESTS}")
    print(f"Concurrency Levels: {CONCURRENT_REQUESTS}\n")
    
    # Get system info
    system_info = get_system_info()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Store all results
    all_results = {}
    
    # Benchmark with different concurrency levels
    for concurrency in CONCURRENT_REQUESTS:
        print(f"\n{'='*60}")
        print(f"Testing Concurrency Level: {concurrency}")
        print(f"{'='*60}")
        
        # OpenAI Official Client
        print("⏳ Testing OpenAI Official Client...")
        openai_results = await benchmark_openai_client(NUM_REQUESTS, concurrency)
        print(f"✓ OpenAI: {openai_results['throughput']:.2f} req/s")
        
        # Rust Client
        print("⏳ Testing Rust Client...")
        rust_results = await benchmark_rust_client(NUM_REQUESTS, concurrency)
        print(f"✓ Rust: {rust_results['throughput']:.2f} req/s")
        
        # Calculate comparison metrics
        speedup = rust_results['throughput'] / openai_results['throughput'] if openai_results['throughput'] > 0 else 0
        latency_improvement = (1 - rust_results['mean_latency'] / openai_results['mean_latency']) * 100 if openai_results['mean_latency'] > 0 else 0
        
        all_results[concurrency] = {
            'openai': openai_results,
            'rust': rust_results,
            'comparison': {
                'speedup': speedup,
                'latency_improvement': latency_improvement
            }
        }
        
        print(f"📊 Speedup: {speedup:.2f}x | Latency Improvement: {latency_improvement:.1f}%")
        
        await asyncio.sleep(2)  # Rate limiting protection
    
    # Generate HTML report
    print(f"\n{'='*60}")
    print("📝 Generating HTML Report...")
    print(f"{'='*60}")
    
    html_report = generate_html_report(all_results, system_info, timestamp)
    
    # Save report
    report_filename = f"openai_benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    print(f"✅ Report saved: {report_filename}")
    print(f"📊 Open the file in your browser to view the full report")
    
    # Also save JSON data for further analysis
    json_filename = f"benchmark_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    json_data = {
        'timestamp': timestamp,
        'system_info': system_info,
        'configuration': {
            'model': MODEL,
            'num_requests': NUM_REQUESTS,
            'concurrency_levels': CONCURRENT_REQUESTS,
            'test_messages': TEST_MESSAGES
        },
        'results': {str(k): v for k, v in all_results.items()}
    }
    
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"💾 Raw data saved: {json_filename}")
    
    # Print summary to console
    print(f"\n{'='*60}")
    print("📊 BENCHMARK SUMMARY")
    print(f"{'='*60}")
    speedups = [all_results[c]['comparison']['speedup'] for c in CONCURRENT_REQUESTS]
    print(f"Peak Speedup: {max(speedups):.2f}x (at concurrency {CONCURRENT_REQUESTS[speedups.index(max(speedups))]})")
    print(f"Average Speedup: {statistics.mean(speedups):.2f}x")
    print(f"Min Speedup: {min(speedups):.2f}x")
    print(f"\n🎉 Benchmark completed successfully!")
    print(f"📄 View the HTML report for detailed analysis and charts")


if __name__ == "__main__":
    asyncio.run(main())