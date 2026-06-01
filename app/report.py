from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)
from reportlab.lib.units import cm
import io
from datetime import datetime


def generate_pdf(result, inputs):
    """Generate a downloadable PDF report for the prediction."""
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(buffer, pagesize=A4,
                               rightMargin=2*cm, leftMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    story  = []

    # Risk color
    risk_color_map = {
        'LOW RISK':    colors.HexColor('#27AE60'),
        'MEDIUM RISK': colors.HexColor('#F39C12'),
        'HIGH RISK':   colors.HexColor('#E74C3C')
    }
    rc = risk_color_map.get(result['risk_label'],
                             colors.HexColor('#3498DB'))

    # Header
    story.append(Paragraph(
        "RiskSense AI — Loan Default Risk Report",
        ParagraphStyle('Title', fontSize=20, fontName='Helvetica-Bold',
                       spaceAfter=6, textColor=colors.HexColor('#2C3E50'))
    ))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}",
        ParagraphStyle('Sub', fontSize=10, textColor=colors.gray,
                       spaceAfter=4)
    ))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor('#BDC3C7')))
    story.append(Spacer(1, 0.4*cm))

    # Risk verdict
    story.append(Paragraph(
        f"Risk Assessment: {result['risk_emoji']} {result['risk_label']}",
        ParagraphStyle('Risk', fontSize=16, fontName='Helvetica-Bold',
                       textColor=rc, spaceAfter=4)
    ))
    story.append(Paragraph(
        f"Default Probability: {result['probability']}%",
        ParagraphStyle('Prob', fontSize=13, spaceAfter=12,
                       textColor=colors.HexColor('#2C3E50'))
    ))
    story.append(Spacer(1, 0.3*cm))

    # Applicant details table
    story.append(Paragraph("Applicant Details",
        ParagraphStyle('H2', fontSize=13, fontName='Helvetica-Bold',
                       spaceAfter=6,
                       textColor=colors.HexColor('#2C3E50'))))

    table_data = [['Feature', 'Value']] + [
        [k, str(round(v, 3))] for k, v in inputs.items()
    ]
    t = Table(table_data, colWidths=[10*cm, 6*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,0), colors.HexColor('#2C3E50')),
        ('TEXTCOLOR',   (0,0), (-1,0), colors.white),
        ('FONTNAME',    (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',    (0,0), (-1,-1), 10),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.HexColor('#F8F9FA'), colors.white]),
        ('GRID',        (0,0), (-1,-1), 0.5,
         colors.HexColor('#BDC3C7')),
        ('PADDING',     (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5*cm))

    # Top SHAP factors
    story.append(Paragraph("Top Factors Driving This Prediction",
        ParagraphStyle('H2', fontSize=13, fontName='Helvetica-Bold',
                       spaceAfter=6,
                       textColor=colors.HexColor('#2C3E50'))))

    shap_data = [['Feature', 'Impact', 'Direction']]
    for _, row in result['shap_df'].head(6).iterrows():
        direction = '↑ Increases Risk' if row['SHAP Value'] > 0 \
                    else '↓ Decreases Risk'
        shap_data.append([
            row['Feature'],
            f"{abs(row['SHAP Value']):.4f}",
            direction
        ])

    st = Table(shap_data, colWidths=[8*cm, 4*cm, 5*cm])
    st.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,0), colors.HexColor('#2C3E50')),
        ('TEXTCOLOR',   (0,0), (-1,0), colors.white),
        ('FONTNAME',    (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',    (0,0), (-1,-1), 10),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.HexColor('#F8F9FA'), colors.white]),
        ('GRID',        (0,0), (-1,-1), 0.5,
         colors.HexColor('#BDC3C7')),
        ('PADDING',     (0,0), (-1,-1), 6),
    ]))
    story.append(st)
    story.append(Spacer(1, 0.5*cm))

    # Footer
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor('#BDC3C7')))
    story.append(Paragraph(
        "RiskSense AI | Built with Random Forest + SHAP | "
        "MSc Data Science Portfolio Project",
        ParagraphStyle('Footer', fontSize=8,
                       textColor=colors.gray, spaceBefore=6)
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer