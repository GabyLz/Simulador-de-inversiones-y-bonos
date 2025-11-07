"""
UI para el conversor de monedas en Streamlit
Componente intuitivo para mostrar tasas en tiempo real,
hacer conversiones y permitir fallback manual.

Incluye:
- Widget conversor interactivo
- Historial de conversiones en session_state
- Auditoría de tasas usadas
- Manejo de errores con fallback
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from modules.fx_converter import (
    get_fx_rate,
    convert_currency,
    get_supported_currencies,
    is_valid_currency,
    UnsupportedCurrencyError,
    ProviderError,
    RateNotFoundError,
    FXCache,
)

def show_fx_converter_widget():
    """
    Widget principal del conversor de monedas.
    Intuitivo, con manejo de errores y fallback manual.
    """
    
    # Inicializar cache global en session_state
    if 'fx_cache' not in st.session_state:
        st.session_state.fx_cache = FXCache()

    # Inicializar valores de monedas si no existen
    if 'fx_from' not in st.session_state:
        st.session_state.fx_from = 'PEN'

    if 'fx_to' not in st.session_state:
        st.session_state.fx_to = 'USD'

    if 'fx_amount' not in st.session_state:
        st.session_state.fx_amount = 100.0
    
    if 'last_conversion' not in st.session_state:
        st.session_state.last_conversion = None
    
    st.subheader("💱 Conversor de Monedas (Tiempo Real)")
    st.caption("Tasas actualizadas automáticamente desde open.er-api.com (API gratuita, sin autenticación)")
    
    # Obtener lista de monedas soportadas
    currencies = get_supported_currencies()
    
    # Dividir en dos columnas para entrada de datos
    col1, col2, col3 = st.columns([1, 0.5, 1])
    
    with col1:
        amount = st.number_input(
            "Monto a convertir",
            min_value=0.0,
            value=st.session_state.fx_amount,
            step=0.01,
            format="%.2f"
        )
        st.session_state.fx_amount = amount
    
    with col2:
        st.write("")  # Espacio visual
        st.write("")
        if st.button("↔️ Invertir", help="Intercambia moneda origen y destino"):
            # Intercambiar en session_state
            temp = st.session_state.fx_from
            st.session_state.fx_from = st.session_state.fx_to
            st.session_state.fx_to = temp
            st.rerun()
    
    with col3:
        st.write("")  # Espacio visual
    
    # Segunda fila: selección de monedas
    col1_m, col_arrow, col2_m = st.columns([1, 0.3, 1])
    
    with col1_m:
        idx_from = currencies.index(st.session_state.fx_from) if st.session_state.fx_from in currencies else 0
        from_currency = st.selectbox(
            "De (moneda origen)",
            options=currencies,
            index=idx_from
        )
        st.session_state.fx_from = from_currency
    
    with col_arrow:
        st.write("")
        st.write("→")
    
    with col2_m:
        idx_to = currencies.index(st.session_state.fx_to) if st.session_state.fx_to in currencies else 1
        to_currency = st.selectbox(
            "A (moneda destino)",
            options=currencies,
            index=idx_to
        )
        st.session_state.fx_to = to_currency
    
    # Tercera fila: opciones avanzadas (fallback manual)
    with st.expander("⚙️ Opciones avanzadas", expanded=False):
        use_manual_rate = st.checkbox(
            "Usar tasa manual (fallback si API falla)",
            value=False,
            help="Si la API no responde, puedes proporcionar una tasa manual."
        )
        
        manual_rate = None
        if use_manual_rate:
            st.caption(f"Especifica cuánto vale 1 {from_currency} en {to_currency}")
            manual_rate = st.number_input(
                "Tasa manual",
                min_value=0.0001,
                value=1.0,
                step=0.01,
                format="%.6f"
            )
        
        # Opción para limpiar cache
        if st.button("🔄 Limpiar cache de tasas (refrescar)"):
            st.session_state.fx_cache.clear()
            st.success("✅ Cache limpiada. Las tasas se refrescarán.")
    
    # Botón principal: Convertir
    st.markdown("---")
    
    if st.button("🔄 Convertir", use_container_width=True):
        try:
            # Realizar conversión
            result = convert_currency(
                amount,
                from_currency,
                to_currency,
                cache=st.session_state.fx_cache,
                manual_rate=manual_rate if use_manual_rate else None
            )
            
            # Guardar como última conversión
            st.session_state.last_conversion = {
                'timestamp': result['timestamp'],
                'amount_original': result['amount_original'],
                'from_currency': result['from_currency'],
                'to_currency': result['to_currency'],
                'amount_converted': result['amount_converted'],
                'rate': result['rate'],
                'source': result['source'],
                'provider': result['provider']
            }
            
            # Mostrar resultado con estilo
            st.success("✅ Conversión completada")
            
            # Mostrar resultado grande
            col_result_1, col_result_2 = st.columns(2)
            
            with col_result_1:
                st.metric(
                    f"Monto en {from_currency}",
                    f"{result['amount_original']:,.2f}",
                )
            
            with col_result_2:
                st.metric(
                    f"Monto en {to_currency}",
                    f"{result['amount_converted']:,.2f}",
                    delta=f"Tasa: {result['rate']:.6f}"
                )
            
            # Detalles de la conversión
            with st.expander("📊 Detalles de la conversión", expanded=True):
                details_col1, details_col2 = st.columns(2)
                
                with details_col1:
                    st.write(f"**Tasa utilizada:** {result['rate']:.6f}")
                    st.write(f"**Fuente:** {result['source'].upper()}")
                    st.write(f"**Proveedor:** {result['provider']}")
                
                with details_col2:
                    st.write(f"**Timestamp:** {result['timestamp']}")
                    if result['source'] == 'manual':
                        st.info("ℹ️ Usando tasa manual (fallback)")
                    elif result['source'] == 'cache':
                        st.info("ℹ️ Tasa en cache (puede tener 1h de antigüedad)")
                    else:
                        st.success("ℹ️ Tasa en tiempo real")
                
                # Fórmula de cálculo
                st.markdown(f"**Fórmula:**")
                st.code(f"{result['amount_original']} {from_currency} × {result['rate']:.6f} = {result['amount_converted']:.2f} {to_currency}")
        
        except UnsupportedCurrencyError as e:
            st.error(f"❌ Moneda no soportada: {e}")
        except RateNotFoundError as e:
            st.error(f"❌ Tasa no disponible: {e}")
        except ProviderError as e:
            st.error(f"❌ Error del proveedor: {e}")
            st.info("💡 Intenta de nuevo en unos momentos, o usa la opción de tasa manual.")
        except Exception as e:
            st.error(f"❌ Error inesperado: {e}")
    
    # Mostrar historial de conversiones
    st.markdown("---")
    st.subheader("� Panel de Tasas de Monedas Relevantes")
    
    # Selector de moneda de referencia
    col_ref, _ = st.columns([1, 3])
    with col_ref:
        reference_currency = st.selectbox(
            "Ver tasas respecto a:",
            options=currencies,
            index=currencies.index('USD') if 'USD' in currencies else 0,
            key="fx_reference"
        )
    
    # Monedas relevantes para mostrar (prioritariamente latinoamericanas)
    relevant_currencies = ['PEN', 'ARS', 'BRL', 'CLP', 'COP', 'MXN', 'UYU', 'EUR', 'GBP', 'JPY']
    # Filtrar solo las que existen en la lista soportada y excluir la moneda de referencia
    relevant_currencies = [c for c in relevant_currencies if c in currencies and c != reference_currency]
    
    if relevant_currencies:
        st.write(f"**Conversión: 1 {reference_currency} = ?**")
        
        # Crear grid de tasas
        cols = st.columns(min(5, len(relevant_currencies)))  # Máximo 5 columnas
        
        for idx, target_currency in enumerate(relevant_currencies):
            with cols[idx % len(cols)]:
                try:
                    # Obtener tasa
                    rate_result = get_fx_rate(
                        reference_currency,
                        target_currency,
                        cache=st.session_state.fx_cache
                    )
                    
                    rate = rate_result['rate']
                    source = rate_result['source']
                    
                    # Mostrar con emoji según la fuente
                    source_emoji = "🔴" if source == 'api' else "🟡" if source == 'cache' else "🟢"
                    
                    st.metric(
                        f"{target_currency}",
                        f"{rate:.4f}",
                        delta=source_emoji,
                        delta_color="off"
                    )
                except Exception as e:
                    st.warning(f"❌ {target_currency}: {str(e)[:30]}")
        
        # Leyenda de colores
        with st.expander("ℹ️ Leyenda de fuente de tasas"):
            st.caption("🔴 Rojo = Tasa en tiempo real (API)")
            st.caption("🟡 Amarillo = Tasa en cache (última hora)")
            st.caption("🟢 Verde = Tasa manual (fallback)")
    
    # Mostrar última conversión realizada (si existe)
    if 'last_conversion' in st.session_state and st.session_state.last_conversion:
        st.markdown("---")
        st.subheader("📝 Última Conversión Realizada")
        
        last = st.session_state.last_conversion
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Monto Original", f"{last['amount_original']:,.2f} {last['from_currency']}")
        with col2:
            st.metric("Tasa Usada", f"{last['rate']:.6f}")
        with col3:
            st.metric("Resultado", f"{last['amount_converted']:,.2f} {last['to_currency']}")
        
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.caption(f"**Fuente:** {last['source'].upper()}")
        with col_info2:
            st.caption(f"**Hora:** {last['timestamp']}")

