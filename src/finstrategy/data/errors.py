class MarketDataError(Exception):
    """Dado indisponível na fonte de mercado (Yahoo Finance, MT5, ...).

    Não é um bug: ações sem cadeia de opções (ex. a maioria das B3 `.SA`) ou sem
    histórico suficiente caem aqui e ficam marcadas como "não elegíveis", com o motivo.
    """
