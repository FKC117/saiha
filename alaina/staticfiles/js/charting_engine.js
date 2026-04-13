/**
 * ChatFlow Advanced Charting Engine (Phase 7)
 * Implements a robust lifecycle for ECharts with ResizeObserver and stable resurrection.
 */

const ChatChartManager = (() => {
    // Registry for active charts and their observers
    const charts = new Map();
    const observers = new Map();

    // Zinc-950 / Glassmorphism Theming
    const ZincTheme = {
        color: ['#8B5CF6', '#6366F1', '#A78BFA', '#4F46E5', '#C084FC', '#818CF8'],
        textStyle: { color: '#D4D4D8', fontFamily: 'Inter, system-ui, sans-serif' },
        title: { textStyle: { color: '#F4F4F5', fontWeight: '600', fontSize: 14 } },
        grid: { borderColor: 'rgba(255,255,255,0.05)', containLabel: true },
        categoryAxis: {
            axisLine: { lineStyle: { color: '#3F3F46' } },
            axisTick: { show: false },
            splitLine: { show: false },
            axisLabel: { color: '#A1A1AA' }
        },
        valueAxis: {
            axisLine: { show: false },
            axisTick: { show: false },
            splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)', type: 'dashed' } },
            axisLabel: { color: '#A1A1AA' }
        },
        tooltip: {
            backgroundColor: 'rgba(24, 24, 27, 0.9)',
            borderColor: '#3F3F46',
            borderWidth: 1,
            textStyle: { color: '#F4F4F5' },
            borderRadius: 8,
            shadowBlur: 10,
            shadowColor: 'rgba(0,0,0,0.5)'
        }
    };

    /**
     * Transforms normalized data intent into an ECharts option object.
     * Prevents version lock-in by keeping logic in JS.
     */
    function buildOption(type, data) {
        let option = {
            title: { text: data.title || '', left: 'left' },
            tooltip: { trigger: (type === 'scatter' || type === 'boxplot') ? 'item' : 'axis' },
            legend: { show: !!data.legend, bottom: 0, textStyle: { color: '#A1A1AA' } },
            grid: { left: '3%', right: '4%', bottom: '15%', top: '15%', containLabel: true },
            toolbox: { feature: { saveAsImage: { show: true, type: 'png', backgroundColor: '#18181b' } } }
        };

        const meta = data.metadata || {};

        switch (type.toLowerCase()) {
            case 'line':
            case 'area':
                option.xAxis = { type: 'category', data: data.xAxis || [] };
                option.yAxis = { type: 'value', name: meta.yAxisLabel || '' };
                option.series = (data.series || []).map(s => ({
                    name: s.name,
                    type: 'line',
                    smooth: true,
                    areaStyle: type === 'area' ? { opacity: 0.1 } : null,
                    data: s.data
                }));
                break;

            case 'bar':
                option.xAxis = { type: 'category', data: data.xAxis || [] };
                option.yAxis = { type: 'value', name: meta.yAxisLabel || '' };
                option.series = (data.series || []).map(s => ({
                    name: s.name,
                    type: 'bar',
                    borderRadius: [4, 4, 0, 0],
                    data: s.data
                }));
                break;

            case 'boxplot':
                option.xAxis = { type: 'category', data: data.categories || [] };
                option.yAxis = { type: 'value', name: meta.yAxisLabel || '' };
                option.series = [
                    {
                        name: 'boxplot',
                        type: 'boxplot',
                        data: data.values || [],
                        itemStyle: { borderColor: '#8B5CF6', borderWidth: 2, color: 'rgba(139, 92, 246, 0.2)' }
                    }
                ];
                break;

            case 'heatmap':
                option.xAxis = { type: 'category', data: data.x || [] };
                option.yAxis = { type: 'category', data: data.y || [] };
                option.visualMap = {
                    min: data.min || 0,
                    max: data.max || 1,
                    calculable: true,
                    orient: 'horizontal',
                    left: 'center',
                    bottom: '0%',
                    inRange: { color: ['#3F3F46', '#8B5CF6', '#F4F4F5'] }
                };
                option.series = [{
                    name: data.title || 'Correlation',
                    type: 'heatmap',
                    data: data.values || [],
                    label: { show: true, fontSize: 10, color: '#A1A1AA' },
                    emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0, 0, 0, 0.5)' } }
                }];
                break;

            case 'scatter':
                option.xAxis = { type: 'value', name: meta.xAxisLabel || '' };
                option.yAxis = { type: 'value', name: meta.yAxisLabel || '' };
                option.series = (data.series || []).map(s => ({
                    name: s.name,
                    type: 'scatter',
                    symbolSize: val => Math.sqrt(val[2] || 10) * 2,
                    data: s.data
                }));
                break;

            case 'pie':
            case 'donut':
                option.series = [{
                    name: data.title || '',
                    type: 'pie',
                    radius: type === 'donut' ? ['40%', '70%'] : '70%',
                    avoidLabelOverlap: false,
                    itemStyle: { borderRadius: 8, borderColor: '#18181b', borderWidth: 2 },
                    label: { show: false, position: 'center' },
                    emphasis: { label: { show: true, fontSize: '14', fontWeight: 'bold' } },
                    data: data.values || []
                }];
                break;

            default:
                console.warn(`Unsupported chart type: ${type}`);
        }

        return option;
    }

    /**
     * Entry point: Initialize a persistent chart with stable ResizeObserver lifecycle.
     */
    function initChart(id, container, type, data) {
        if (!container || !echarts) return null;

        // 1. Clean up existing if ID collision (Safe Resurrection)
        if (charts.has(id)) {
            destroyChart(id);
        }

        // 2. Initialize instance
        const chart = echarts.init(container);
        
        // 3. Selection: Manual Transformation (Legacy) or Pass-Through (Hardened)
        // If data has 'series' but no 'type', or if we detect the hardened 'option' pattern
        let finalOption = data;
        if (type && type !== 'RAW') {
             finalOption = buildOption(type, data);
        }
        
        // 4. Apply custom Zinc theme overrides + Final Option
        chart.setOption({...ZincTheme, ...finalOption});

        // 4. Setup Robust Resize Management
        const observer = new ResizeObserver(() => {
            // Check if element is still in DOM and visible
            if (container.clientWidth > 0) {
                chart.resize();
            }
        });
        observer.observe(container);

        // 5. Register for lifecycle tracking
        charts.set(id, chart);
        observers.set(id, observer);

        return chart;
    }

    /**
     * Memory-safe destruction of a single chart and its observer.
     */
    function destroyChart(id) {
        if (charts.has(id)) {
            charts.get(id).dispose();
            charts.delete(id);
        }
        if (observers.has(id)) {
            observers.get(id).disconnect();
            observers.delete(id);
        }
    }

    /**
     * Destroys all registered charts (Used during session switching).
     */
    function destroyAll() {
        console.log(`[ChatChartManager] Cleaning up ${charts.size} chart instances...`);
        charts.forEach((_, id) => destroyChart(id));
        charts.clear();
        observers.clear();
    }

    return {
        initChart,
        destroyChart,
        destroyAll
    };
})();

// Attach to window for global access (though usually used via exports in modular apps)
window.ChatChartManager = ChatChartManager;
