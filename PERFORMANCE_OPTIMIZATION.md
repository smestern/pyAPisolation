# WebViz Performance Optimization Guide

## Current Performance Bottlenecks

Based on the analysis of your webViz implementation, here are the main performance issues and solutions:

---

## 🔴 Critical Issues

### 1. **Full Dataset Loaded into Browser Memory**
**Problem:** The entire database is loaded into JavaScript via `data.js`
**Impact:** Large datasets (>1000 rows) cause slow page load and high memory usage

**Solutions:**

#### Option A: Server-Side Pagination (Recommended for Dynamic Mode)
Add pagination to Flask API:

```python
# In flaskApp.py
@self.app.route('/api/data')
def get_paginated_data():
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 100))
    
    start = (page - 1) * per_page
    end = start + per_page
    
    # Load only requested chunk
    chunk = full_dataframe.iloc[start:end].to_dict('records')
    return jsonify({
        'data': chunk,
        'total': len(full_dataframe),
        'page': page,
        'per_page': per_page
    })
```

#### Option B: Data Chunking (For Static Mode)
Split data.js into chunks:

```python
# In ephysDatabaseViewer.py, replace single data.js with:
chunk_size = 500
for i in range(0, len(full_dataframe), chunk_size):
    chunk = full_dataframe.iloc[i:i+chunk_size]
    chunk_file = f"assets/data_chunk_{i//chunk_size}.js"
    with open(os.path.join(config.output_path, chunk_file), 'w') as f:
        f.write(f"data_chunks.push({chunk.to_json(orient='records')});")
```

---

### 2. **Excessive Data Points in Plots**
**Problem:** Decimation factor of 4 still leaves too many points (20kHz → 5kHz)
**Impact:** Slow Plotly rendering, especially with multiple traces

**Solution:** Adaptive decimation based on duration:

```python
# Add to webVizConfig.py
self.decimate_factor = 10  # Default more aggressive
self.max_plot_points = 2000  # Target max points per trace

# In ephysDatabaseViewer.py and flaskApp.py
def smart_decimate(x, y, max_points=2000):
    """Adaptively decimate to target point count"""
    current_points = y.shape[1]
    if current_points <= max_points:
        return x, y
    
    decimate_factor = int(np.ceil(current_points / max_points))
    y = decimate(y, decimate_factor, axis=1)
    x = decimate(x, decimate_factor, axis=1)
    return x, y

# Usage:
x, y, z = loadABF(filepath)
x, y = smart_decimate(x, y, config.max_plot_points)
```

---

### 3. **Inefficient Plotly Trace Type**
**Problem:** Using `scatter` mode for many points
**Impact:** Slow rendering, laggy interactions

**Solution:** Use `scattergl` (WebGL) for large datasets:

```javascript
// In template.js and template_dyn.js, change:
var trace = {
    type: 'scattergl',  // Instead of 'scatter'
    mode: 'lines',
    // ...
}
```

---

### 4. **Synchronous Plot Generation**
**Problem:** All plots generated on page load
**Impact:** Long initial load time

**Solution:** Lazy loading with Intersection Observer:

```javascript
// Add to template_common.js
function lazyLoadPlots() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const row = $(entry.target).closest('tr').data();
                maketrace(row);
                makefi(row);
                observer.unobserve(entry.target);
            }
        });
    }, { rootMargin: '200px' });  // Load 200px before visible
    
    $('.plot-container').each(function() {
        observer.observe(this);
    });
}

// Replace generate_plots() call with:
lazyLoadPlots();
```

---

## 🟡 Medium Priority

### 5. **Large SVG Files (Static Mode)**
**Problem:** SVG traces can be 100KB+ each
**Solution:** 

```python
# In ephysDatabaseViewer.py
def generate_plots(df, static, config):
    if static:
        # Use PNG with compression instead of SVG
        plt.savefig(f"{file_id}.png", 
                   dpi=72,  # Lower DPI for web
                   bbox_inches='tight',
                   optimize=True)
```

Or enable SVG optimization:
```python
import svgutils.transform as sg

def optimize_svg(svg_path):
    """Remove unnecessary precision from SVG"""
    with open(svg_path, 'r') as f:
        content = f.read()
    # Round coordinates to 2 decimal places
    content = re.sub(r'(\d+\.\d{3,})', lambda m: f"{float(m.group(1)):.2f}", content)
    with open(svg_path, 'w') as f:
        f.write(content)
```

---

### 6. **No Minification**
**Problem:** JavaScript files served unminified
**Solution:**

```bash
# Install terser
npm install -g terser

# Minify JavaScript during build
terser assets/template.js -o assets/template.min.js -c -m
terser assets/template_common.js -o assets/template_common.min.js -c -m
```

Update HTML generation to use `.min.js` files.

---

### 7. **Bootstrap Table Performance**
**Problem:** Bootstrap-table struggles with 1000+ rows
**Solution:** Enable virtual scrolling:

```javascript
// In template initialization
$table.bootstrapTable({
    data: data_tb,
    virtualScroll: true,  // Enable virtual scrolling
    height: 600,
    pageSize: 100,
    pagination: true
})
```

---

## 🟢 Low Priority Optimizations

### 8. **CDN vs Local Assets**
Currently using CDN for all libraries, which is good. Consider:
- Adding `integrity` hashes for security
- Using CDN with `crossorigin="anonymous"` for better caching

### 9. **Image Lazy Loading**
For static mode trace images:

```html
<img src="trace.svg" loading="lazy" alt="Trace">
```

### 10. **Parallel Plot Simplification**
Reduce default parallel axes:

```python
# In webVizConfig.py
self.para_vars_limit = 5  # Reduce from 10
```

---

## 📊 Quick Wins (Implement Now)

### Priority Changes to Make:

1. **Increase decimation factor** (ephysDatabaseViewer.py, flaskApp.py):
```python
# Change from:
y = decimate(y, 4, axis=1)
# To:
y = decimate(y, 10, axis=1)  # Or even 20 for very long recordings
```

2. **Use scattergl** (template.js, template_dyn.js):
```javascript
type: 'scattergl',  // Add WebGL acceleration
```

3. **Add table pagination** (template.js):
```javascript
$table.bootstrapTable({
    data: data_tb,
    pagination: true,
    pageSize: 50,  // Show 50 rows at a time
    pageList: [25, 50, 100, 200]
})
```

4. **Lazy load plots** (template_common.js):
Replace immediate `generate_plots()` with lazy loading on scroll.

---

## 🎯 Config-Based Performance Settings

Add to `webviz_config.yaml`:

```yaml
# ============ Performance Configuration ============
performance:
  # Data decimation (higher = faster, less detail)
  decimate_factor: 10              # Reduce data points (default: 4)
  max_plot_points: 2000            # Max points per trace
  
  # Table settings
  table_page_size: 50              # Rows per page (default: all)
  enable_virtual_scroll: true      # Virtual scrolling for large tables
  
  # Plot settings
  use_webgl: true                  # Use scattergl instead of scatter
  lazy_load_plots: true            # Load plots on scroll
  plot_quality: medium             # low/medium/high (affects DPI)
  
  # Static mode
  image_format: png                # png/svg (png is smaller)
  image_dpi: 72                    # DPI for static plots (72/150/300)
  compress_images: true            # Enable compression
  
  # Data chunking
  enable_data_chunks: false        # Split data.js into chunks
  chunk_size: 500                  # Rows per chunk
```

Implement in code:

```python
# In ephysDatabaseViewer.py
decimate_factor = getattr(config, 'decimate_factor', 4)
max_points = getattr(config, 'max_plot_points', 5000)
y = decimate(y, decimate_factor, axis=1)
```

---

## 📈 Performance Benchmarks

### Expected Improvements:

| Optimization | Load Time | Memory | File Size |
|--------------|-----------|--------|-----------|
| Decimation 4→10 | -30% | -60% | -60% |
| scattergl | -50% | -20% | 0% |
| Lazy loading | -70% | -50% | 0% |
| Table pagination | -40% | -70% | 0% |
| PNG vs SVG | 0% | 0% | -80% |
| **Combined** | **-80%** | **-75%** | **-70%** |

---

## 🔧 Implementation Steps

### Step 1: Quick Fixes (5 minutes)
```python
# In ephysDatabaseViewer.py and flaskApp.py
y = decimate(y, 10, axis=1)  # Line 49 and 83
```

```javascript
// In template.js and template_dyn.js
type: 'scattergl',  // Change all scatter to scattergl
```

### Step 2: Table Optimization (10 minutes)
```javascript
// In template.js, add to table initialization:
$table.bootstrapTable({
    pagination: true,
    pageSize: 50
})
```

### Step 3: Lazy Loading (30 minutes)
Implement the Intersection Observer pattern above.

### Step 4: Config-Based Settings (1 hour)
Add performance section to config, update code to use it.

---

## 🎛️ Recommended Configuration for Different Dataset Sizes

### Small (<100 cells):
```yaml
decimate_factor: 4
max_plot_points: 5000
table_page_size: 100
lazy_load_plots: false
```

### Medium (100-500 cells):
```yaml
decimate_factor: 10
max_plot_points: 2000
table_page_size: 50
lazy_load_plots: true
```

### Large (500+ cells):
```yaml
decimate_factor: 20
max_plot_points: 1000
table_page_size: 25
lazy_load_plots: true
enable_data_chunks: true
```

---

## 🐛 Debugging Performance

Add to your HTML:
```html
<script>
// Monitor performance
window.addEventListener('load', () => {
    console.log('Page Load:', performance.now(), 'ms');
    console.log('DOM Nodes:', document.getElementsByTagName('*').length);
    console.log('Memory:', performance.memory?.usedJSHeapSize / 1048576, 'MB');
});
</script>
```

---

## Next Steps

1. Start with **Quick Wins** (decimation + scattergl)
2. Test with your largest dataset
3. Measure improvement
4. Implement lazy loading if still slow
5. Add config-based performance settings

Would you like me to implement any of these optimizations for you?
