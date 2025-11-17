# LLM Insights Module V2 - Gemini API Integration

## 📋 Tổng quan

Module LLM Insights đã được cập nhật để sử dụng **Gemini API** với prompt template V2 chuyên nghiệp, cung cấp insights chi tiết và actionable cho demand forecasting.

## ✨ Tính năng mới

- ✅ **Prompt V2**: Template chuyên nghiệp với cấu trúc rõ ràng
- ✅ **Gemini API Integration**: Sử dụng Google Gemini Pro làm LLM chính
- ✅ **Rule-based Fallback**: Tự động fallback nếu không có API key
- ✅ **Structured Output**: Format chuẩn với Executive Summary, Causal Factors, Business Impact, Actions
- ✅ **SHAP Integration**: Phân tích feature importance từ SHAP values

## 📁 Files

### 1. `src/modules/llm_prompts.py` (NEW)
Chứa prompt template V2 với các placeholders:
- Product overview
- Forecast metrics (Q05, Q50, Q95)
- Trend analysis
- SHAP feature importance
- Inventory situation
- Risk assessment

### 2. `src/modules/llm_insights.py` (UPDATED)
Module chính với:
- `LLMInsightGenerator` class
- `_format_prompt()` method để format prompt V2
- `_call_gemini_api()` method để gọi Gemini API
- Rule-based fallback với helper methods đầy đủ

## 🚀 Cách sử dụng

### 1. Rule-based Mode (Không cần API)

```python
from src.modules.llm_insights import LLMInsightGenerator

generator = LLMInsightGenerator(use_llm_api=False)

forecast_data = {
    'q50': 150,
    'q05': 100,
    'q95': 200,
    'vs_yesterday': 15.5,
    'vs_last_week': 8.2,
    'current_inventory': 120,
    'safety_stock': 30,
    'reorder_point': 100,
    'stockout_risk_pct': 45,
    'overstock_risk_pct': 20,
    'category': 'Fresh Produce',
    'date': '2025-11-16',
    'horizon': '24 hours'
}

shap_data = {
    'promo_active': 0.35,
    'price_change': -0.15,
    'day_of_week': 0.10
}

insight = generator.generate_forecast_insight(
    'P001',
    forecast_data,
    shap_data
)

print(insight['insight_text'])
```

### 2. Gemini API Mode (Cần API key)

#### Setup API Key:

```powershell
# Windows PowerShell
$env:GEMINI_API_KEY="your-api-key-here"

# Hoặc tạo file .env
echo "GEMINI_API_KEY=your-api-key-here" > .env
```

#### Sử dụng:

```python
from src.modules.llm_insights import LLMInsightGenerator
import os

# Option 1: Từ environment variable
generator = LLMInsightGenerator(
    use_llm_api=True,
    api_provider="gemini",
    model="gemini-2.5-flash"  # Default model
)

# Option 2: Truyền trực tiếp
generator = LLMInsightGenerator(
    api_key="your-api-key",
    use_llm_api=True,
    api_provider="gemini",
    model="gemini-2.5-flash"  # or "gemini-2.5-pro" for better quality
)

insight = generator.generate_forecast_insight(
    'P001',
    forecast_data,
    shap_data,
    use_llm=True  # Force LLM mode
)

print(insight['insight_text'])
```

### 3. Convenience Function

```python
from src.modules.llm_insights import generate_insight

insight_text = generate_insight(
    'P001',
    forecast_data,
    shap_data,
    use_llm=True,
    api_key="your-key",
    api_provider="gemini"
)
# Note: Default model is "gemini-2.5-flash"

print(insight_text)
```

## 📊 Output Format

### Rule-based Output:

```
## 📊 EXECUTIVE SUMMARY

Demand forecast for this product is showing **moderate growth** (+16%) 
with **moderate uncertainty**. Expected demand: **150 units**. 
inventory levels are manageable.

## 🔍 CAUSAL FACTORS

1. **Active Promotion** (35.0% impact)
   - Active promotional campaign driving sales
2. **Yesterday's Demand** (25.0% impact)
   - Yesterday's demand was higher than usual, boosting today's forecast

## 📈 BUSINESS IMPACT

- **Inventory Status**: Below forecast level (80.0%)
- **Stockout Risk**: MODERATE (45%) - Monitor closely

## ✅ RECOMMENDED ACTIONS

1. **👁️ ONGOING - Monitor Key Indicators**
   - Track hourly sales vs forecast
   - Alert if actual demand deviates >20% from Q50
```

### LLM Output (Gemini):

Gemini sẽ tạo insights chi tiết hơn với:
- Executive Summary (2-3 câu)
- Causal Explanation (3-4 bullet points)
- Business Impact Assessment
- Recommended Actions (Priority-ordered)
- Risk Mitigation (nếu cần)

## 🔧 Configuration

### InsightConfig Parameters:

- `use_llm_api`: `bool` - Bật/tắt LLM API
- `api_provider`: `str` - "gemini", "openai", "anthropic"
- `api_key`: `str` - API key (optional, có thể dùng env var)
- `model`: `str` - Model name (default: "gemini-2.5-flash")

### Supported Models:

**Gemini:**
- `gemini-2.5-flash` (recommended - fast and cost-effective)
- `gemini-2.5-pro` (better quality, slower)
- `gemini-pro-latest` (backward compatible)

**OpenAI (fallback):**
- `gpt-4`
- `gpt-3.5-turbo`

**Anthropic (fallback):**
- `claude-3-opus`
- `claude-3-sonnet`

## 📝 Forecast Data Structure

```python
forecast_data = {
    # Required
    'q50': float,              # Median forecast
    'q05': float,              # Pessimistic case
    'q95': float,              # Optimistic case
    
    # Optional but recommended
    'vs_yesterday': float,      # % change vs yesterday
    'vs_last_week': float,      # % change vs last week
    'vs_monthly_avg': float,    # % change vs monthly avg
    
    # Inventory metrics
    'current_inventory': float,
    'safety_stock': float,
    'reorder_point': float,
    'stockout_risk_pct': float,  # 0-100
    'overstock_risk_pct': float, # 0-100
    
    # Metadata
    'category': str,
    'date': str,                # YYYY-MM-DD
    'horizon': str              # e.g., "24 hours"
}
```

## 🧪 Testing

### Test Rule-based:

```bash
python scripts/test_llm_insights_v2.py
```

### Test với Gemini API:

```powershell
# Set API key
$env:GEMINI_API_KEY="your-key"

# Run test
python scripts/test_llm_insights_v2.py
```

## 🔄 Migration từ Version cũ

Nếu bạn đang dùng version cũ:

1. **Import path không đổi**: `from src.modules.llm_insights import LLMInsightGenerator`
2. **API thay đổi**: 
   - Cũ: `InsightConfig(use_llm_api=True, api_provider="openai")`
   - Mới: `LLMInsightGenerator(use_llm_api=True, api_provider="gemini")`
3. **Prompt tự động**: Prompt V2 được sử dụng tự động khi có `llm_prompts.py`

## ⚠️ Lưu ý

1. **API Key Security**: 
   - Không commit API key vào Git
   - Sử dụng environment variables
   - Thêm `.env` vào `.gitignore`

2. **Fallback Behavior**:
   - Nếu không có API key → tự động dùng rule-based
   - Nếu API call fail → tự động fallback về rule-based

3. **Cost Management**:
   - Gemini API có free tier (generous)
   - Monitor usage trong Google Cloud Console
   - Rule-based mode hoàn toàn free

4. **Encoding Issues**:
   - Windows console có thể không hiển thị emoji
   - Code đã xử lý fallback tự động

## 📚 Examples

Xem thêm examples trong:
- `scripts/test_llm_insights_v2.py`
- `scripts/test_gemini_insights.py` (nếu có)

## 🆘 Troubleshooting

### Lỗi: "No API key found"
→ Set environment variable: `$env:GEMINI_API_KEY="your-key"`

### Lỗi: "google-generativeai not installed"
→ Install: `pip install google-generativeai`

### Lỗi: "UnicodeEncodeError"
→ Đã được xử lý tự động, nếu vẫn lỗi thì check console encoding

### LLM không hoạt động
→ Check API key, network connection, hoặc dùng rule-based mode

---

**Version**: 1.0.0  
**Last Updated**: 2025-11-16  
**Author**: SmartGrocy Team

