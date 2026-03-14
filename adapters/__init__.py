from adapters.financeiq_adapter import normalize as normalize_financeiq
from adapters.finqa_adapter import normalize as normalize_finqa
from adapters.convfinqa_adapter import normalize as normalize_convfinqa
from adapters.tatqa_adapter import normalize as normalize_tatqa
from adapters.fincuge_instruction_adapter import normalize as normalize_fincuge_instruction
from adapters.finance_alpaca_adapter import normalize as normalize_finance_alpaca
from adapters.financial_phrasebank_adapter import normalize as normalize_financial_phrasebank


ALL_ADAPTERS = [
    normalize_financeiq,
    normalize_finqa,
    normalize_convfinqa,
    normalize_tatqa,
    normalize_fincuge_instruction,
    normalize_finance_alpaca,
    normalize_financial_phrasebank,
]
