# Simulador real de inversiones (Acciones y Bonos) — Interfaz Streamlit
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import textwrap
import plotly.graph_objects as go
from openai import OpenAI
import os

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# Importar módulo de histórico (con persistencia en localStorage)
from modules.user_data import (
    init_user_session, 
    save_simulation, 
    show_history_tab,
    get_simulation_count
)

# Importar módulo de comparación con mercado
from modules.market_comparison_ui import show_market_comparison

# Importar módulo de chatbot
from modules.chatbot_assistant import show_chatbot, show_chatbot_compact

# Cliente OpenAI
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
st.set_page_config(page_title="Simulador Real de Inversiones", layout="wide")

# -----------------------------
# Helper functions (comunes)
# -----------------------------
def generar_pdf(nombre, edad_actual, edad_final, resultados, tipo, grafico_bytes=None):
    """
    Genera un PDF con formato profesional y colores personalizados.
    - tipo: "Inversiones" o "Bonos" (cambia el color del encabezado)
    """
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.pagesizes import letter

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=40, rightMargin=40, topMargin=50, bottomMargin=40)
    styles = getSampleStyleSheet()

    # --- Paleta de colores por tipo ---
    if tipo.lower().startswith("bon"):
        color_principal = colors.HexColor("#065F46")  # verde financiero
        titulo_texto = "📈 Reporte de Bonos "
    else:
        color_principal = colors.HexColor("#1E3A8A")  # azul profesional
        titulo_texto = "📊 Reporte de Acciones "

    contenido = []

    # Encabezado colorido
    encabezado = Table(
        [[titulo_texto]],
        colWidths=[450]
    )
    encabezado.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color_principal),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 16),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
    ]))
    contenido.append(encabezado)
    contenido.append(Spacer(1, 15))

    # Datos personales
    contenido.append(Paragraph(f"<b>Nombre:</b> {nombre}", styles["Normal"]))
    contenido.append(Paragraph(f"<b>Edad actual:</b> {edad_actual} años", styles["Normal"]))
    contenido.append(Paragraph(f"<b>Edad al finalizar inversión:</b> {edad_final} años", styles["Normal"]))
    contenido.append(Spacer(1, 15))

    # Resultados (como tabla)
    data_tabla = [["Concepto", "Valor"]]
    for k, v in resultados.items():
        data_tabla.append([k, str(v)])
    tabla = Table(data_tabla, colWidths=[200, 200])
    tabla.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), color_principal),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
    ]))
    contenido.append(tabla)
    contenido.append(Spacer(1, 20))

    # Gráfico si existe
    if grafico_bytes:
        contenido.append(Paragraph("<b>Evolución de la inversión</b>", styles["Heading3"]))
        contenido.append(Spacer(1, 6))
        try:
            img = RLImage(BytesIO(grafico_bytes))
            img._restrictSize(440, 250)
            contenido.append(img)
        except Exception:
            contenido.append(Paragraph("<i>No fue posible insertar el gráfico</i>", styles["Italic"]))
        contenido.append(Spacer(1, 15))

    # Pie de página con QR / logo
    qr_path = os.path.join("telegram", "qr_contacto.png")
    if os.path.exists(qr_path):
        try:
            contenido.append(Spacer(1, 20))
            contenido.append(Paragraph("<b>Contacto del asesor financiero:</b>", styles["Normal"]))
            qr_img = RLImage(qr_path)
            qr_img._restrictSize(100, 100)
            contenido.append(qr_img)
        except Exception:
            contenido.append(Paragraph("<i>(QR no disponible)</i>", styles["Italic"]))

    contenido.append(Spacer(1, 10))
    contenido.append(Paragraph(
        "<font size=8 color=grey>Reporte generado automáticamente por el Simulador Real de Inversiones.</font>",
        styles["Normal"]
    ))

    # Generar PDF
    doc.build(contenido)
    buffer.seek(0)
    return buffer

def enviar_email(destinatario, pdf_bytes, nombre_pdf):
    """Envía el PDF por Gmail al usuario."""
    msg = MIMEMultipart()
    msg["From"] = st.secrets["EMAIL_USER"]
    msg["To"] = destinatario
    msg["Subject"] = "📊 Resultados de tu simulación de inversión"

    cuerpo = MIMEText("Adjunto encontrarás el reporte PDF con los resultados de tu simulación.\n\nAtentamente,\nSimulador Real de Inversiones", "plain")
    msg.attach(cuerpo)

    # Adjuntar PDF
    adjunto = MIMEApplication(pdf_bytes.getvalue(), _subtype="pdf")
    adjunto.add_header("Content-Disposition", "attachment", filename=nombre_pdf)
    msg.attach(adjunto)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(st.secrets["EMAIL_USER"], st.secrets["EMAIL_PASS"])
        smtp.send_message(msg)


def validate_inputs(initial, annuity, years, tea_pct, freq):
    errors = []
    if initial is None: initial = 0.0
    if annuity is None: annuity = 0.0
    if initial < 0: errors.append("La inversión inicial no puede ser negativa.")
    if annuity < 0: errors.append("La anualidad no puede ser negativa.")
    if years <= 0: errors.append("El horizonte de inversión (años) debe ser mayor que 0.")
    if not (0 <= tea_pct <= 1000): errors.append("TEA inválida. Debe estar entre 0% y 1000% (revisar).")
    if freq not in ("Mensual", "Trimestral", "Anual"): errors.append("Modalidad inválida.")
    if initial == 0 and annuity == 0: errors.append("Ingrese al menos inversión inicial o anualidad.")
    return errors

def tea_to_periodic(tea_pct, freq):
    tea = tea_pct / 100.0
    if freq == "Mensual":
        r = (1 + tea) ** (1/12) - 1
        per = 12
    elif freq == "Trimestral":
        r = (1 + tea) ** (1/4) - 1
        per = 4
    else:  # Anual
        r = tea
        per = 1
    return r, per

def future_value(initial, annuity, r, n_periods):
    fv_init = initial * ((1 + r) ** n_periods)
    fv_ann = 0.0
    if annuity > 0:
        if r == 0:
            fv_ann = annuity * n_periods
        else:
            fv_ann = annuity * (((1 + r) ** n_periods - 1) / r)
    return fv_init + fv_ann, fv_init, fv_ann

def breakdown_over_time(initial, annuity, r, n_periods):
    balances = []
    bal = initial
    for t in range(1, n_periods + 1):
        bal = bal * (1 + r)
        bal += annuity
        balances.append(bal)
    return balances

# -----------------------------
# Helper functions for BONDS
# -----------------------------
def period_to_name(period):
    mapping = {
        "Mensual": "meses",
        "Trimestral": "trimestres",
        "Semestral": "semestres",
        "Anual": "años"
    }
    return mapping.get(period, "períodos")

def periods_per_year(period):
    mapping = {
        "Mensual": 12,
        "Trimestral": 4,
        "Semestral": 2,
        "Anual": 1
    }
    return mapping[period]

def bond_present_value(face_value, coupon_rate_annual, yield_tea, period, n_periods):
    """
    Calcula el valor presente de un bono.
    - face_value: valor nominal
    - coupon_rate_annual: tasa cupón anual (%)
    - yield_tea: tasa de rendimiento requerida (TEA, %)
    - period: periodicidad
    - n_periods: número total de períodos
    """
    n_per_year = periods_per_year(period)
    coupon_rate_eff = (1 + coupon_rate_annual / 100) ** (1/n_per_year) - 1
    yield_eff = (1 + yield_tea / 100) ** (1/n_per_year) - 1

    coupon_payment = face_value * coupon_rate_eff

    # Valor presente de cupones
    if yield_eff == 0:
        pv_coupons = coupon_payment * n_periods
    else:
        pv_coupons = coupon_payment * (1 - (1 + yield_eff) ** (-n_periods)) / yield_eff

    # Valor presente del valor nominal
    pv_face = face_value / ((1 + yield_eff) ** n_periods)

    total_pv = pv_coupons + pv_face
    return total_pv, coupon_payment, coupon_rate_eff, yield_eff

# -----------------------------
# UI: Diseño & Tema
# -----------------------------
st.markdown("""
<style>
[data-testid='column'] > div:first-child {padding: 8px}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Ajustes de diseño ✨")
    theme = st.selectbox(
        "Tema",
        ["Claro (default)", "Verde - Energía", "Azul - Profesional", "Minimal"]
    )
    
    # Guardar tema en session_state para otros módulos
    st.session_state.current_theme = theme
    
    font_size = st.slider("Tamaño de letra (px)", 12, 20, 14)
    compact = st.checkbox("Compactar espacios", value=False)
    
    st.markdown("---")
    
    # Chatbot rápido en sidebar
    show_chat_sidebar = st.checkbox("💬 Chat rápido IA", value=False)
    if show_chat_sidebar:
        show_chatbot_compact()
    
    st.markdown("---")


# -----------------------------
# 💅 Aplicar estilos globales dinámicos
# -----------------------------
padding_value = '8px' if compact else '18px'

# Solo aplicar estilos si NO está en el modo "default"
if theme != "Claro (default)":
    if theme == "Verde - Energía":
        bg_color = "#F0FFF4"
        text_color = "#065F46"
        accent_color = "#10B981"
        sidebar_bg = "#A1D2B4"
    elif theme == "Azul - Profesional":
        bg_color = "#F0F9FF"
        text_color = "#1E3A8A"
        accent_color = "#2563EB"
        sidebar_bg = "#E8F0FF"
    elif theme == "Minimal":
        # 🎨 Tono oscuro suave con buen contraste
        bg_color = "#F5F5F5"
        text_color = "#2C2C2C"
        accent_color = "#606060"
        sidebar_bg = "#2F2F2F"  # gris oscuro elegante

    # CSS global dinámico (Dashboard + Sidebar)
    st.markdown(f"""
    <style>
    html, body, [class*="css"] {{
        font-size: {font_size}px !important;
        background-color: {bg_color} !important;
        color: {text_color} !important;
    }}
    .stApp {{
        padding: {padding_value};
        background-color: {bg_color};
        color: {text_color};
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {accent_color} !important;
    }}
    .stButton>button {{
        background-color: {accent_color};
        color: white !important;
        border-radius: 10px;
        border: none;
    }}
    .stButton>button:hover {{
        background-color: {text_color};
        color: {bg_color} !important;
        border: 1px solid {accent_color};
    }}
    hr {{
        border: 1px solid {accent_color};
    }}

    /* === Sidebar completo === */
    section[data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        color: {text_color} !important;
        padding: 20px;
        border-radius: 0 10px 10px 0;
        box-shadow: 2px 0 10px rgba(0,0,0,0.1);
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] h5,
    section[data-testid="stSidebar"] h6,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {{
        color: {"#FFFFFF" if theme == "Minimal" else text_color} !important;
    }}
    section[data-testid="stSidebar"] .stImage img {{
        border-radius: 8px;
        border: 2px solid {accent_color};
    }}
    </style>
    """, unsafe_allow_html=True)
# -----------------------------
# Main layout
# -----------------------------
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Simulador real de inversiones — ACCIONES Y BONOS")
    st.write("Simula fondos de inversión (Acciones) o valores de deuda (Bonos).")
with col2:
    st.write("")
    st.caption("Desarrollado con cariño — Grupo 09")

st.markdown("---")

# Inicializar sesión de histórico
init_user_session()

tab_acciones, tab_bonos, tab_historico, tab_chatbot = st.tabs([
    "💰 Acciones", 
    "📈 Bonos", 
    "📜 Mi Histórico",
    "💬 Chatbot IA"
])

# =================
# PESTAÑA ACCIONES (sin cambios)
# =================
with tab_acciones:
    from modules.presets import get_preset_acciones, list_presets_acciones
    
    # Selector de presets COMPACTO
    col_preset_label, col_preset_select = st.columns([0.3, 0.7])
    with col_preset_label:
        st.markdown("**📋 Plantillas:**")
    with col_preset_select:
        preset_options = [("", "✨ Elegir...")] + list_presets_acciones()
        preset_selected = st.selectbox(
            "Plantillas predefinidas",
            options=[opt[0] for opt in preset_options],
            format_func=lambda x: [opt[1] for opt in preset_options if opt[0] == x][0] if x else "✨ Elegir...",
            key="preset_acciones",
            label_visibility="collapsed"
        )
    
    # Si usuario selecciona preset, mostrar info en expander compacto
    if preset_selected and preset_selected != "":
        preset_data = get_preset_acciones(preset_selected)
        if preset_data:
            with st.expander(f"ℹ️ {preset_data['nombre']} - ${preset_data['initial']:,} | {preset_data['tea_pct']}% | {preset_data['years']}Y", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"💰 Inicial: ${preset_data['initial']:,}")
                    st.caption(f"💵 Aporte: ${preset_data['annuity']:,}")
                with col2:
                    st.caption(f"📊 TEA: {preset_data['tea_pct']}%")
                    st.caption(f"⏰ Plazo: {preset_data['years']} años")
                st.caption(preset_data['descripcion'])
    
    with st.form("form_acciones"):
        # Pre-llenar valores si se eligió preset
        if preset_selected and preset_selected != "":
            preset_data = get_preset_acciones(preset_selected)
            if preset_data:
                default_initial = preset_data["initial"]
                default_annuity = preset_data["annuity"]
                default_years = preset_data["years"]
                default_tea = preset_data["tea_pct"]
                default_modality = preset_data["modality"]
            else:
                default_initial, default_annuity, default_years, default_tea, default_modality = 10000, 0, 5, 12, "Mensual"
        else:
            default_initial, default_annuity, default_years, default_tea, default_modality = 10000, 0, 5, 12, "Mensual"
        
        # Información del inversionista (agregado)
        st.markdown("### Información del inversionista")
        nombre = st.text_input("Nombre completo", help="Tu nombre para personalizar el reporte.")
        edad = st.number_input("Edad actual", min_value=18, max_value=100, value=30, help="Edad que tienes actualmente.")
        correo = st.text_input("Correo electrónico", help="A este correo se enviará el PDF con tus resultados.")

        st.subheader("Parámetros de la inversión")
        r1, r2, r3 = st.columns([1,1,1])
        with r1:
            initial = st.number_input("Inversión inicial ($)", min_value=0.0, value=float(default_initial), step=100.0, format="%.2f")
            annuity = st.number_input("Anualidad / aporte periódico (en la modalidad elegida)", min_value=0.0, value=float(default_annuity), step=50.0, format="%.2f")
        with r2:
            years = st.number_input("Tiempo total (años)", min_value=0.1, value=float(default_years), step=0.5)
            tea_pct = st.number_input("TEA (%) — tasa anual efectiva", min_value=0.0, value=float(default_tea), step=0.1, format="%.4f")
        with r3:
            modality = st.selectbox("Modalidad de la anualidad", ["Mensual", "Trimestral", "Anual"], index=["Mensual", "Trimestral", "Anual"].index(default_modality))
            bolsa = st.selectbox("Selecciona la bolsa (afecta impuesto)", ["BOLSA LOCAL (5%)", "BOLSA EXTRANJERA (29.5%)"]) 
        st.markdown("**Opciones de cálculo**")
        colA, colB = st.columns(2)
        with colA:
            withdraw_mode = st.radio("Modo de salida de ganancia", ("Retirar todo (utilidad pura)", "Retirar dividendos (mantener capital)"))
        with colB:
            dividend_override = st.checkbox("Personalizar tasa de dividendos (por defecto = TEA/2)")
            if dividend_override:
                dividend_pct = st.number_input("Tasa de dividendos anual (%)", value=tea_pct/2.0, min_value=0.0)
            else:
                dividend_pct = None
        submitted = st.form_submit_button("Calcular")
    
    # Realizar cálculos si se envió el formulario
    if submitted:
        errors = validate_inputs(initial, annuity, years, tea_pct, modality)
        if errors:
            st.error("\n".join(errors))
        else:
            r_period, per_year = tea_to_periodic(tea_pct, modality)
            n_periods = int(np.round(years * per_year))
            fv_total, fv_init, fv_ann = future_value(initial, annuity, r_period, n_periods)
            tax_rate = 0.05 if "LOCAL" in bolsa else 0.295
            total_invested = initial + annuity * n_periods
            gain_before_tax = fv_total - total_invested
            gain_before_tax = max(gain_before_tax, 0.0)
            tax_on_withdrawal = gain_before_tax * tax_rate
            net_gain_withdrawal = gain_before_tax - tax_on_withdrawal

            div_pct = (dividend_pct if dividend_pct is not None else tea_pct / 2.0) / 100.0
            annual_dividend = fv_total * div_pct
            monthly_dividend = annual_dividend / 12.0
            net_monthly_dividend = monthly_dividend * (1 - tax_rate)
            net_annual_dividend = annual_dividend * (1 - tax_rate)
            total_net_dividends_over_period = net_annual_dividend * years

            # Guardar TODOS los datos necesarios en session_state
            st.session_state.update({
                "initial": initial,
                "annuity": annuity,
                "years": years,
                "tea_pct": tea_pct,
                "modality": modality,
                "bolsa": bolsa,
                "nombre": nombre,
                "correo": correo,
                "edad": edad,
                "dividend_pct": dividend_pct,
                "fv_total": fv_total,
                "fv_init": fv_init,
                "fv_ann": fv_ann,
                "tax_rate": tax_rate,
                "total_invested": total_invested,
                "gain_before_tax": gain_before_tax,
                "tax_on_withdrawal": tax_on_withdrawal,
                "net_gain_withdrawal": net_gain_withdrawal,
                "div_pct": div_pct,
                "annual_dividend": annual_dividend,
                "monthly_dividend": monthly_dividend,
                "net_monthly_dividend": net_monthly_dividend,
                "net_annual_dividend": net_annual_dividend,
                "total_net_dividends_over_period": total_net_dividends_over_period,
                "r_period": r_period,
                "per_year": per_year,
                "n_periods": n_periods
            })
    
    # Mostrar resultados si existen datos en session_state
    if "fv_total" in st.session_state and st.session_state.fv_total is not None:
        # Recuperar todas las variables desde session_state
        initial = st.session_state.initial
        annuity = st.session_state.annuity
        years = st.session_state.years
        tea_pct = st.session_state.tea_pct
        modality = st.session_state.modality
        bolsa = st.session_state.bolsa
        nombre = st.session_state.get("nombre", "")
        correo = st.session_state.get("correo", "")
        edad = st.session_state.get("edad", 0)
        dividend_pct = st.session_state.get("dividend_pct")
        fv_total = st.session_state.fv_total
        fv_init = st.session_state.fv_init
        fv_ann = st.session_state.fv_ann
        tax_rate = st.session_state.tax_rate
        total_invested = st.session_state.total_invested
        gain_before_tax = st.session_state.gain_before_tax
        tax_on_withdrawal = st.session_state.tax_on_withdrawal
        net_gain_withdrawal = st.session_state.net_gain_withdrawal
        div_pct = st.session_state.div_pct
        annual_dividend = st.session_state.annual_dividend
        monthly_dividend = st.session_state.monthly_dividend
        net_monthly_dividend = st.session_state.net_monthly_dividend
        net_annual_dividend = st.session_state.net_annual_dividend
        total_net_dividends_over_period = st.session_state.total_net_dividends_over_period
        r_period = st.session_state.r_period
        per_year = st.session_state.per_year
        n_periods = st.session_state.n_periods

        st.subheader("Resultados (resumen)")
        colr1, colr2 = st.columns(2)
        with colr1:
            st.metric("Valor futuro acumulado (FV)", f"$ {fv_total:,.2f}")
            st.write("- Aporte inicial futuro:", f"$ {fv_init:,.2f}")
            st.write("- Aportes periódicos futuros:", f"$ {fv_ann:,.2f}")
            st.write("- Invertido neto (suma aportes): ", f"$ {total_invested:,.2f}")
        with colr2:
            st.metric("Impuesto aplicado", f"{tax_rate*100:.2f}%")
            st.write("- Ganancia antes de impuestos (si retira todo):", f"$ {gain_before_tax:,.2f}")
            st.write("- Impuesto si retira todo:", f"$ {tax_on_withdrawal:,.2f}")
            st.write("- Ganancia neta tras retiro total:", f"$ {net_gain_withdrawal:,.2f}")
        st.markdown("---")
        st.subheader("Escenario: Retirar dividendos (no retirar capital)")
        st.write(f"Tasa de dividendos anual usada: {div_pct*100:.4f}%")
        st.write(f"Dividendo anual (bruto): $ {annual_dividend:,.2f}")
        st.write(f"Dividendo anual (neto después de impuestos): $ {net_annual_dividend:,.2f}")
        st.write(f"Dividendo mensual neto estimado: $ {net_monthly_dividend:,.2f}")
        st.write(f"Total dividendos netos en {years} años (sin tocar capital): $ {total_net_dividends_over_period:,.2f}")

        st.markdown("---")
        st.subheader("Comparación rápida: ¿Qué te conviene más?")
        comp_df = pd.DataFrame({
            "Escenario": ["Retirar todo (una vez)", "Retirar dividendos (sumados en período)"],
            "Ganancia neta ($)": [net_gain_withdrawal, total_net_dividends_over_period]
        })
        st.table(comp_df)

        st.subheader("Evolución del fondo en el tiempo")
        # Calculamos balances actuales
        balances = breakdown_over_time(initial, annuity, r_period, n_periods)
        df_bal = pd.DataFrame({
            "Periodo": np.arange(1, n_periods + 1),
            "Balance": balances
        })

        # Guardamos el gráfico anterior en sesión para comparación
        if "last_chart_df" not in st.session_state:
            st.session_state.last_chart_df = None
        if "last_tea" not in st.session_state:
            st.session_state.last_tea = None

        # Crear figura Plotly
        fig = go.Figure()

        # Si hay un gráfico anterior, lo mostramos en gris con su TEA
        if st.session_state.last_chart_df is not None:
            fig.add_trace(go.Scatter(
                x=st.session_state.last_chart_df["Periodo"],
                y=st.session_state.last_chart_df["Balance"],
                mode='lines',
                name=f"TEA = {st.session_state.last_tea:.2f}% (anterior)",
                line=dict(color='gray', dash='dash', width=2),
                opacity=0.6,
                hovertemplate="Periodo %{x}<br>Balance $ %{y:,.2f}<extra></extra>"
            ))

        # Línea actual destacada
        fig.add_trace(go.Scatter(
            x=df_bal["Periodo"],
            y=df_bal["Balance"],
            mode='lines',
            name=f"TEA = {tea_pct:.2f}% (actual)",
            line=dict(color='royalblue', width=3),
            hovertemplate="Periodo %{x}<br>Balance $ %{y:,.2f}<extra></extra>"
        ))

        # Configuración estética del gráfico
        fig.update_layout(
            width=800,
            height=400,
            xaxis_title="Periodo",
            yaxis_title="Balance ($)",
            template="plotly_white",
            legend=dict(x=0, y=1.1, orientation="h"),
            margin=dict(l=40, r=20, t=40, b=40)
        )

        # Mostrar gráfico interactivo en Streamlit
        st.plotly_chart(
            fig,
            config={
                "responsive": True,     # hace que el gráfico se adapte al tamaño del contenedor
                "displaylogo": False,   # quita el logo de Plotly
                "scrollZoom": True      # permite hacer zoom con la rueda del ratón
            },
            use_container_width=True    # (opcional) hace que el gráfico ocupe todo el ancho del contenedor
        )

        # Actualizar datos en sesión
        st.session_state.last_chart_df = df_bal.copy()
        st.session_state.last_tea = tea_pct

        st.markdown("---")
        # --- SECCIÓN DE DESCARGA Y DETALLES ---
        st.markdown("### 📊 Descarga y detalles")

        # 💄 CSS mínimo solo para alinear y dar buen espaciado
        st.markdown("""
        <style>
        div[data-testid="stHorizontalBlock"] div[data-testid="stVerticalBlock"] {
            align-items: center !important;
        }
        .stButton>button, .stDownloadButton>button {
            border-radius: 8px;
            height: 42px;
            font-weight: 500;
        }
        </style>
        """, unsafe_allow_html=True)

        # 🔹 Tres columnas en la misma línea
        col1, col2, col3 = st.columns([1.5, 1, 2])

        # 📥 Columna 1: Botón CSV
        with col1:
            csv = df_bal.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Descargar CSV",
                data=csv,
                file_name='evolucion_fondo.csv',
                mime='text/csv',
                use_container_width=True
            )

        # 🧮 Columna 2: Edad final
        with col2:
            edad_final = edad + years
            st.markdown(
                f"""
                <div style='
                    text-align:center;
                    background-color:#f0f4ff;
                    padding:10px 0;
                    border-radius:8px;
                    border:1px solid #d0d8f0;
                    font-weight:600;
                '>
                    🧮 Edad final: {edad_final:.0f} años
                </div>
                """,
                unsafe_allow_html=True
            )

        # ✉️ Columna 3: Campo correo y botón juntos
        with col3:
            if correo and correo.strip():
                grafico_bytes = None
                try:
                    grafico_bytes = fig.to_image(format="png")
                except Exception:
                    try:
                        fig_m, ax_m = plt.subplots(figsize=(8, 3.5))
                        ax_m.plot(df_bal["Periodo"], df_bal["Balance"], linewidth=2)
                        ax_m.set_xlabel("Periodo")
                        ax_m.set_ylabel("Balance ($)")
                        ax_m.grid(True, linestyle='--', alpha=0.6)
                        buf_fb = BytesIO()
                        fig_m.tight_layout()
                        fig_m.savefig(buf_fb, format="png")
                        plt.close(fig_m)
                        buf_fb.seek(0)
                        grafico_bytes = buf_fb.getvalue()
                    except Exception as e_img:
                        st.warning(f"⚠️ No fue posible generar la imagen del gráfico: {e_img}")
                        grafico_bytes = None

                resultados_accion = {
                    "Valor futuro acumulado": f"$ {fv_total:,.2f}",
                    "Ganancia neta total": f"$ {net_gain_withdrawal:,.2f}",
                    "Dividendo anual neto": f"$ {net_annual_dividend:,.2f}",
                    "Tipo de bolsa": bolsa,
                    "TEA (%)": f"{tea_pct:.2f}",
                }

                pdf_buffer = generar_pdf(
                    nombre or "N/A",
                    int(edad or 0),
                    int(edad_final),
                    resultados_accion,
                    'Acciones',
                    grafico_bytes
                    
                )

                if st.button("📤 Enviar PDF por correo", use_container_width=True):
                    try:
                        enviar_email(correo, pdf_buffer, "Resultados_Acciones.pdf")
                        st.success(f"✅ Se envió el PDF con los resultados a **{correo}**")
                    except Exception as e:
                        st.error(f"❌ No se pudo enviar el correo: {e}")
            else:
                st.warning("⚠️ Introduce un correo válido para enviar el PDF.")

        # ---- Auto-guardar simulación en histórico ----
        # Solo guardar si acabamos de calcular (evitar duplicados al rerun por botones)
        if submitted:
            params_dict = {
                "nombre": nombre,
                "correo": correo,
                "edad": edad,
                "inicial": initial,
                "anualidad": annuity,
                "years": years,
                "tea_pct": tea_pct,
                "modalidad": modality,
                "bolsa": bolsa
            }
            results_dict = {
                "fv_total": fv_total,
                "ganancia_neta_retiro": net_gain_withdrawal,
                "dividendo_anual_neto": net_annual_dividend,
                "total_dividendos_periodo": total_net_dividends_over_period
            }
            save_simulation("Acciones", params_dict, results_dict, auto_save=True)
        
        # Comparación con mercado real (Yahoo Finance) - siempre mostrar si hay datos
        
        
        # Mensaje si se cargó desde histórico
        if st.session_state.get("loaded_from_history", False):
            st.success("✅ **Simulación cargada desde el histórico** - Ahora puedes comparar con el mercado")
            st.session_state.loaded_from_history = False  # Reset flag
        
        show_market_comparison(tea_pct, years, initial, fv_total=fv_total)

    st.markdown("---")
    st.subheader("Recomendaciones con IA")
    if "ia_response" not in st.session_state:
        st.session_state.ia_response = None
    if "ia_loading" not in st.session_state:
        st.session_state.ia_loading = False

    if st.button("💡 Pedir recomendación a la IA"):
        # Validación: evitar llamar a la API si no hay datos de la simulación
        missing_fields = []
        # Campos principales que deben existir/ser no nulos
        if not st.session_state.get("initial") and not st.session_state.get("annuity"):
            missing_fields.append("Aporte inicial o aportes periódicos")
        if not st.session_state.get("years"):
            missing_fields.append("Años de inversión")
        if not st.session_state.get("tea_pct"):
            missing_fields.append("TEA")
        # Valor futuro estimado (resultado de la simulación) también es importante
        fv = st.session_state.get("fv_total")

        if missing_fields or fv is None or (isinstance(fv, (int, float)) and fv <= 0):
            # Mensaje claro al usuario indicando qué debe completar/ejecutar
            if fv is None or (isinstance(fv, (int, float)) and fv <= 0):
                st.warning("Primero ejecuta la simulación o introduce datos válidos (valor futuro o parámetros de inversión).")
            if missing_fields:
                st.warning("Faltan datos: " + ", ".join(missing_fields))
        else:
            st.session_state.ia_loading = True
            st.session_state.ia_response = None
            try:
                prompt = textwrap.dedent(f"""
                Eres un asesor financiero profesional.
                Analiza el siguiente caso:
                - Inversión inicial: {st.session_state.initial} PEN
                - Aportes periódicos: {st.session_state.annuity} cada {st.session_state.modality}
                - Duración total: {st.session_state.years} años
                - TEA: {st.session_state.tea_pct}%
                - Valor futuro estimado: {st.session_state.fv_total:.2f} PEN
                - Tipo de bolsa: {st.session_state.bolsa}
                Tu tarea:
                1️⃣ Explica si conviene retirar todo o quedarse con dividendos.  
                2️⃣ Sugiere un % razonable de dividendos si se desea personalizarlo.  
                3️⃣ Recomienda una simple estrategia de diversificación (máx 3 puntos).  
                Responde en español, con tono profesional pero cercano.
                """)
                with st.spinner("Analizando inversión con IA... ⏳"):
                    completion = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=800,
                    )
                st.session_state.ia_response = completion.choices[0].message.content
                st.session_state.ia_loading = False
            except Exception as e:
                st.session_state.ia_response = f"⚠️ Error al llamar a la API: {e}"
                st.session_state.ia_loading = False
                st.rerun()

    if st.session_state.ia_loading:
        st.info("Analizando inversión con IA... ⏳")
    if st.session_state.ia_response:
        st.success("✅ Recomendación generada:")
        st.markdown(f"> {st.session_state.ia_response}")

# =================
# PESTAÑA BONOS 
# =================
with tab_bonos:
    from modules.presets import get_preset_bonos, list_presets_bonos
    from modules.bond_comparables import get_risk_assessment, BOND_COMPARABLES
    
    st.subheader("Calculadora de Bonos ($)")
    
    # Selector de presets para bonos COMPACTO
    col_preset_label_b, col_preset_select_b = st.columns([0.3, 0.7])
    with col_preset_label_b:
        st.markdown("**📋 Plantillas:**")
    with col_preset_select_b:
        preset_options_bonos = [("", "✨ Elegir...")] + list_presets_bonos()
        preset_selected_bonos = st.selectbox(
            "Plantillas de bonos predefinidas",
            options=[opt[0] for opt in preset_options_bonos],
            format_func=lambda x: [opt[1] for opt in preset_options_bonos if opt[0] == x][0] if x else "✨ Elegir...",
            key="preset_bonos",
            label_visibility="collapsed"
        )
    
    # Si usuario selecciona preset, mostrar info en expander compacto
    if preset_selected_bonos and preset_selected_bonos != "":
        preset_data_bonos = get_preset_bonos(preset_selected_bonos)
        if preset_data_bonos:
            with st.expander(f"ℹ️ {preset_data_bonos['nombre']} - ${preset_data_bonos['face_value']:,} | {preset_data_bonos['coupon_rate']}% | {preset_data_bonos['tea_yield']}%", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"💰 Valor nominal: ${preset_data_bonos['face_value']:,}")
                    st.caption(f"📊 Cupón: {preset_data_bonos['coupon_rate']}%")
                with col2:
                    st.caption(f"💵 TEA requerida: {preset_data_bonos['tea_yield']}%")
                    st.caption(f"⏰ Plazo: {preset_data_bonos['years']} años")
                st.caption(preset_data_bonos['descripcion'])
                st.write(f"- Plazo: {preset_data_bonos['n_periods']} períodos")
    
    st.markdown("---")

    with st.form("form_bonos"):
        # Pre-llenar valores si se eligió preset de bonos
        if preset_selected_bonos and preset_selected_bonos != "":
            preset_data_bonos = get_preset_bonos(preset_selected_bonos)
            if preset_data_bonos:
                default_face_value = preset_data_bonos["face_value"]
                default_coupon_rate = preset_data_bonos["coupon_rate"]
                default_tea_yield = preset_data_bonos["tea_yield"]
                default_period = preset_data_bonos["period"]
                default_n_periods = preset_data_bonos["n_periods"]
            else:
                default_face_value, default_coupon_rate, default_tea_yield, default_period, default_n_periods = 100000, 8.0, 8.0, "Semestral", 20
        else:
            default_face_value, default_coupon_rate, default_tea_yield, default_period, default_n_periods = 100000, 10.0, 12.0, "Semestral", 20
        
        # Información del inversionista (agregado)
        st.markdown("### Información del inversionista")
        nombre = st.text_input("Nombre completo (Bono)", help="Tu nombre para personalizar el reporte.", key="nombre_b")
        edad = st.number_input("Edad actual (Bono)", min_value=18, max_value=100, value=30, help="Edad que tienes actualmente.", key="edad_b")
        correo = st.text_input("Correo electrónico (Bono)", help="A este correo se enviará el PDF con tus resultados.", key="correo_b")

        c1, c2 = st.columns(2)
        with c1:
            face_value = st.number_input("Valor nominal ($)", min_value=1.0, value=float(default_face_value), step=1000.0, format="%.2f")
            period = st.selectbox("Periodicidad del cupón", ["Mensual", "Trimestral", "Semestral", "Anual"], index=["Mensual", "Trimestral", "Semestral", "Anual"].index(default_period))
        with c2:
            time_input = st.number_input(f"Tiempo total ({period_to_name(period)})", min_value=1, value=int(default_n_periods), step=1)
            coupon_rate = st.number_input("Tasa cupón anual (%)", min_value=0.0, value=float(default_coupon_rate), step=0.1, format="%.4f")

        tea_yield = st.number_input("Tasa de rendimiento requerida (TEA, %)", min_value=0.0, value=float(default_tea_yield), step=0.1, format="%.4f")

        # Alerta dinámica (sin form submit)
        st.markdown("### Evaluación de tasa")
        if tea_yield < coupon_rate:
            st.markdown('<span style="color:red;">⚠️ TEA < TASA DE BONO (BONO SOBRE LA PAR).</span>', unsafe_allow_html=True)
        elif tea_yield > coupon_rate:
            st.markdown('<span style="color:green;">TEA > TASA DE BONO (BONO BAJO LA PAR).</span>', unsafe_allow_html=True)
        else:
            st.info("TEA = TASA DE BONO (BONO A LA PAR)")

        submitted_bond = st.form_submit_button("Calcular valor del bono")

    if submitted_bond:
        n_periods = int(time_input)
        try:
            pv, coupon_payment, coupon_eff, yield_eff = bond_present_value(
                face_value, coupon_rate, tea_yield, period, n_periods
            )

            st.session_state.update({
                "bond_face_value": face_value,
                "bond_coupon_rate": coupon_rate,
                "bond_tea_yield": tea_yield,
                "bond_period": period,
                "bond_n_periods": n_periods,
                "bond_pv": pv,
                "bond_coupon_payment": coupon_payment
            })

            st.subheader("Resultados del bono")
            st.metric("Precio justo del bono (valor presente)", f"$ {pv:,.2f}")
            st.write(f"- Cupón periódico: $ {coupon_payment:,.2f}")
            st.write(f"- Número de períodos: {n_periods} ({period_to_name(period)})")
            st.write(f"- Valor nominal recibido al final: $ {face_value:,.2f}")

            # --- NUEVA SECCIÓN: Comparación con mercado ---
            st.markdown("---")
            st.subheader("📊 Comparación con mercado")
            
            with st.expander("🔍 Análisis de competitividad de tu TEA", expanded=True):
                # Obtener análisis de riesgo
                risk_analysis = get_risk_assessment(tea_yield, coupon_rate, int(n_periods / periods_per_year(period)))
                
                # Mostrar clasificación
                col_risk1, col_risk2 = st.columns(2)
                with col_risk1:
                    st.markdown(f"**Tu TEA requerida:** {tea_yield}%")
                    st.markdown(f"**Clasificación:** {risk_analysis['icon']}")
                with col_risk2:
                    st.markdown(f"**Evaluación de plazo:** {risk_analysis['plazo_evaluation']}")
                    st.markdown(f"**Evaluación de cupón:** {risk_analysis['coupon_evaluation']}")
                
                # Comparables más cercanos
                st.markdown("**📈 Bonos comparables en el mercado:**")
                for idx, (key, bond_data, diff) in enumerate(risk_analysis['closest_comparables'], 1):
                    col1, col2, col3 = st.columns([0.4, 0.3, 0.3])
                    with col1:
                        st.caption(f"{idx}. {bond_data['name']}")
                    with col2:
                        st.caption(f"TEA: {bond_data['current_yield']}% | Rating: {bond_data['rating']}")
                    with col3:
                        spread = tea_yield - bond_data['current_yield']
                        if spread < 0:
                            st.caption(f"📉 Tu TEA: {spread:.1f}% menor")
                        else:
                            st.caption(f"📈 Tu TEA: +{spread:.1f}% mayor")
                
                # Recomendación
                st.markdown("---")
                if risk_analysis['classification'] == 'muy_conservador':
                    st.success("✅ Tu TEA es muy competitiva (baja exigencia de retorno)")
                elif risk_analysis['classification'] == 'conservador':
                    st.info("ℹ️ Tu TEA es conservadora (retorno moderado)")
                elif risk_analysis['classification'] == 'realista':
                    st.info("ℹ️ Tu TEA es realista según comparables de mercado")
                elif risk_analysis['classification'] == 'optimista':
                    st.warning("⚠️ Tu TEA es optimista (requiere buen perfil de riesgo)")
                else:
                    st.error("❌ Tu TEA es muy optimista (riesgo significativo)")

            # Flujo de efectivo dentro de un acordeón
            st.markdown("---")
            with st.expander("📊 Ver flujo de efectivo del bono"):
                st.subheader("Flujo de efectivo detallado")
                
                # Generar flujos
                flows = [{"Período": i, "Flujo ($)": coupon_payment} for i in range(1, n_periods)]
                flows.append({"Período": n_periods, "Flujo ($)": coupon_payment + face_value})
                df_flows = pd.DataFrame(flows)

                # Formatear columna de flujo para mostrar con comas y 2 decimales
                df_flows["Flujo ($)"] = df_flows["Flujo ($)"].apply(lambda x: f"${x:,.2f}")
                st.dataframe(df_flows, width='stretch', hide_index=True)
                
                # Resumen
                st.markdown("**Resumen del flujo:**")
                col_f1, col_f2, col_f3 = st.columns(3)
                with col_f1:
                    st.metric("Períodos", n_periods)
                with col_f2:
                    st.metric("Cupón periódico", f"${coupon_payment:,.2f}")
                with col_f3:
                    st.metric("Pago final", f"${coupon_payment + face_value:,.2f}")

                # Gráfico mejorado con matplotlib
                st.markdown("**Visualización del flujo en el tiempo:**")
                fig, ax = plt.subplots(figsize=(10, 4))
                periods = np.arange(1, n_periods + 1)
                cash_flows = [coupon_payment] * (n_periods - 1) + [coupon_payment + face_value]
                
                # Barras normales
                bars = ax.bar(periods[:-1], [coupon_payment] * (n_periods - 1), label="Cupón")
                # Barra final (más destacada)
                ax.bar(periods[-1], coupon_payment + face_value, label="Cupón + Valor nominal")
                
                # Etiquetas en las barras (solo si no son demasiados períodos)
                if n_periods <= 20:
                    for i, v in enumerate(cash_flows):
                        ax.text(i + 1, v + max(cash_flows) * 0.02, f"${v:,.0f}", ha='center', fontsize=9)
                
                ax.set_xlabel("Período")
                ax.set_ylabel("Flujo de efectivo ($)")
                ax.set_title("Flujo de efectivo del bono a lo largo del tiempo")
                ax.legend()
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                ax.set_ylim(0, max(cash_flows) * 1.15)
                
                st.pyplot(fig)

            # ---- Generar PDF + enviar correo (bono) ----

            periodos_por_año = periods_per_year(period)
            años_totales = n_periods / periodos_por_año
            edad_final = edad + años_totales

            unidad_tiempo = period_to_name(period)
            st.info(f"🧮 Edad al finalizar inversión: {edad_final:.1f} años "
                    f"(equivalente a {n_periods} {unidad_tiempo})")

            if correo is None or correo.strip() == "":
                st.warning("Introduce un correo para enviar el PDF (campo vacío).")
            else:
                # Convertir gráfico matplotlib a imagen PNG (si 'fig' existe)
                grafico_bytes = None
                try:
                    buf = BytesIO()
                    fig.savefig(buf, format="png")
                    plt.close(fig)
                    buf.seek(0)
                    grafico_bytes = buf.getvalue()
                except Exception as e_img:
                    st.warning(f"No fue posible generar la imagen del gráfico: {e_img}")
                    grafico_bytes = None

                resultados_bono = {
                    "Precio justo del bono": f"$ {pv:,.2f}",
                    "Cupón periódico": f"$ {coupon_payment:,.2f}",
                    "Períodos totales": str(n_periods),
                    "Tasa cupón anual": f"{coupon_rate:.2f}%",
                    "TEA requerida": f"{tea_yield:.2f}%",
                }

                pdf_buffer = generar_pdf(nombre or "N/A", int(edad or 0), int(np.round(edad_final)), resultados_bono,'bon', grafico_bytes)

                try:
                    enviar_email(correo, pdf_buffer, "Resultados_Bono.pdf")
                    st.success(f"📧 Se envió un PDF con los resultados a {correo}")
                except Exception as e:
                    st.error(f"No se pudo enviar el correo: {e}")

                # ---- Auto-guardar simulación en histórico ----
                params_dict = {
                    "nombre": nombre,
                    "correo": correo,
                    "edad": edad,
                    "valor_nominal": face_value,
                    "tasa_cupon_anual": coupon_rate,
                    "tea_yield": tea_yield,
                    "periodos": n_periods,
                    "periodo_tipo": period
                }
                results_dict = {
                    "bond_pv": pv,
                    "cupon_periodico": coupon_payment,
                    "periodos_totales": n_periods
                }
                save_simulation("Bonos", params_dict, results_dict, auto_save=True)

        except Exception as e:
            st.error(f"Error en cálculo: {e}")
    
    
    # Evaluación clara: ¿Conviene comprar?
    if "bond_pv" in st.session_state:
        st.markdown("---")
        st.subheader("¿Te conviene comprarlo a ese precio?")
        
        tea_yield = st.session_state.bond_tea_yield
        coupon_rate = st.session_state.bond_coupon_rate

        if tea_yield > coupon_rate:
            decision = "✅ Sí, conviene comprar"
            color = "green"
            reason = "La TEA es mayor que la tasa del bono. Lo compras más barato (con descuento) y ganas más rendimiento."
        elif tea_yield < coupon_rate:
            decision = "❌ No conviene comprar"
            color = "red"
            reason = "La TEA es menor que la tasa del bono. Estarías pagando más de lo que vale (a un precio alto o con prima)."
        else:
            decision = "ℹ️ Es indiferente"
            color = "blue"
            reason = "La TEA es igual a la tasa del bono. Se vende a su valor justo (a la par)."

        st.markdown(f'<span style="color:{color}; font-weight:bold;">{decision}</span>', unsafe_allow_html=True)
        st.write(reason)


    # Recomendación IA para bonos
    st.markdown("---")
    st.subheader("Recomendación con IA")
    if "ia_response_bond" not in st.session_state:
        st.session_state.ia_response_bond = None
    if "ia_loading_bond" not in st.session_state:
        st.session_state.ia_loading_bond = False

    if st.button("💡 Pedir recomendación para este bono"):
        if "bond_pv" not in st.session_state:
            st.warning("Primero calcula el valor del bono.")
        else:
            st.session_state.ia_loading_bond = True
            st.session_state.ia_response_bond = None
            try:
                prompt = textwrap.dedent(f"""
                Eres un asesor financiero especializado en renta fija.
                Analiza el siguiente bono:
                - Valor nominal: USD {st.session_state.bond_face_value:,.2f}
                - Tasa cupón anual: {st.session_state.bond_coupon_rate}%
                - Periodicidad: {st.session_state.bond_period}
                - Plazo: {st.session_state.bond_n_periods} {period_to_name(st.session_state.bond_period)}
                - Tasa de rendimiento requerida (TEA): {st.session_state.bond_tea_yield}%
                - Precio justo calculado: USD {st.session_state.bond_pv:,.2f}

                Tu tarea:
                1️⃣ Evalúa si este bono es una buena inversión según las tasas.
                2️⃣ Compara brevemente con alternativas (ej. bonos del tesoro, depósitos).
                3️⃣ Da una recomendación clara: comprar, evitar o considerar con precaución.
                Responde en español, profesional y conciso.
                """)
                with st.spinner("Analizando bono con IA... ⏳"):
                    completion = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=600,
                    )
                st.session_state.ia_response_bond = completion.choices[0].message.content
                st.session_state.ia_loading_bond = False
            except Exception as e:
                st.session_state.ia_response_bond = f"⚠️ Error al llamar a la API: {e}"
                st.session_state.ia_loading_bond = False
                st.rerun()

    if st.session_state.get("ia_loading_bond"):
        st.info("Analizando bono con IA... ⏳")
    if st.session_state.get("ia_response_bond"):
        st.success("✅ Recomendación generada:")
        st.markdown(f"> {st.session_state.ia_response_bond}")

# =================
# PESTAÑA HISTÓRICO
# =================
with tab_historico:
    show_history_tab()

# =================
# PESTAÑA CHATBOT IA
# =================
with tab_chatbot:
    show_chatbot()

# Footer 
st.markdown("---")


st.markdown("""
<div style="
    background-color: #f8f9fa;
    border-radius: 15px;
    padding: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
">
    <div style="flex: 1; color:#333;">
        <h4 style="margin-bottom:5px;">💼 Simulador Real de Inversiones</h4>
        <p style="margin:0; font-size:15px; color:#555;">
            ¿Tienes algún problema o sugerencia? 
            <strong>Comunícate con el área de mantenimiento</strong> 
            escaneando el código QR o escribiéndonos directamente.
        </p>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([4, 1.2])
with col1:
    st.empty()
with col2:
    qr_path = os.path.join("telegram", "qr_contacto.png")
    if os.path.exists(qr_path):
        qr_image = Image.open(qr_path)
        st.image(
            qr_image, 
            width=120, 
            caption="Escanéame 📱", 
            use_container_width=False
        )
    else:
        st.markdown("<p style='color:#888; text-align:center;'>QR no disponible</p>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
