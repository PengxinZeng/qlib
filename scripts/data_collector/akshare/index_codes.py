# Index names are verified online via Sina index quote endpoint.
INDEX_COMPREHENSIVE = [
    "sh.000001",  # 上证指数
    "sz.399106",  # 深证综指
]
INDEX_SIZE = [
    "sh.000016",  # 上证50
    "sh.000300",  # 沪深300
    "sh.000905",  # 中证500
    "sz.399001",  # 深证成指
    "sh.000906",  # 中证800
    "sh.000907",  # 中证700
]
INDEX_INDUSTRY_L1 = [
    "sh.000037",  # 上证医药
    "sh.000038",  # 上证金融
    "sh.000039",  # 上证信息
    "sh.000040",  # 上证电信
    "sh.000041",  # 上证公用
    "sh.000042",  # 上证央企
    "sz.399433",  # 国证交运
    "sz.399434",  # 数字传媒
    "sz.399436",  # 绿色煤炭
]
INDEX_INDUSTRY_L2 = [
    "sh.000952",  # 300地产
    "sz.399951",  # 300银行
    "sh.000913",  # 300医药
    "sh.000914",  # 300金融
    "sh.000915",  # 300信息
    "sh.000916",  # 300电信
    "sh.000917",  # 300公用
]
INDEX_STRATEGY = [
    "sh.000050",  # 50等权
    "sh.000982",  # 500等权
    "sh.000926",  # 中证央企
    "sh.000923",  # 公司债
    "sh.000921",  # 300R价值
    "sh.000922",  # 中证红利
    "sh.000149",  # 180红利
    "sh.000150",  # 380红利
    "sz.399411",  # 红利100
    "sz.399645",  # 100低波
    "sz.399661",  # 深证低波
    "sz.399672",  # 深红利50
    "sz.399692",  # 创业低波
]
INDEX_GROWTH = [
    "sz.399376",  # 小盘成长
    "sz.399377",  # 小盘价值
    "sz.399378",  # ESG 300
]
INDEX_VALUE = [
    "sh.000029",  # 180价值
    "sz.399370",  # 国证成长
    "sh.000030",  # 180R成长
]
INDEX_THEME = [
    "sh.000015",  # 红利指数
    "sh.000063",  # 上证周期
    "sh.000065",  # 上证龙头
    "sh.000066",  # 上证商品
    "sh.000068",  # 上证资源
]
INDEX_FUND = [
    "sh.000011",  # 基金指数
    "sz.399305",  # 基金指数
]
INDEX_BOND = [
    "sh.000012",  # 国债指数
    "sh.000013",  # 企债指数
]

ALL_INDEX_CODES = (
    INDEX_COMPREHENSIVE
    + INDEX_SIZE
    + INDEX_INDUSTRY_L1
    + INDEX_INDUSTRY_L2
    + INDEX_STRATEGY
    + INDEX_GROWTH
    + INDEX_VALUE
    + INDEX_THEME
    + INDEX_FUND
    + INDEX_BOND
)

# Deduplicate while preserving order.
ALL_INDEX_CODES = list(dict.fromkeys(ALL_INDEX_CODES))

# Supported valuation indexes from:
# akshare/stock_feature/stock_a_pe_and_pb.py -> stock_index_pe_lg / stock_index_pb_lg
LG_INDEX_SYMBOL_MAP = {
    "上证50": "000016.SH",
    "沪深300": "000300.SH",
    "上证380": "000009.SH",
    "创业板50": "399673.SZ",
    "中证500": "000905.SH",
    "上证180": "000010.SH",
    "深证红利": "399324.SZ",
    "深证100": "399330.SZ",
    "中证1000": "000852.SH",
    "上证红利": "000015.SH",
    "中证100": "000903.SH",
    "中证800": "000906.SH",
}
LG_MARKET_SYMBOL_MAP = {"上证": "1", "深证": "2", "创业板": "4", "科创版": "7"}
LG_SUPPORTED_INDEX_NAMES = list(LG_INDEX_SYMBOL_MAP.keys()) + list(LG_MARKET_SYMBOL_MAP.keys())
