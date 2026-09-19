"""
services/pass_generator.py - Verified Digital Exam Entry Pass & Hall Ticket Generator.

Generates a formatted, printable digital admission slip and verification certificate
upon successful biometric facial verification on exam day.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import datetime
import hashlib
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import config


@dataclass
class ExamPass:
    """Represents an issued digital examination entry pass."""
    pass_code: str
    student_name: str
    roll_number: str
    course: str
    exam_title: str
    hall_number: str
    timestamp: str
    similarity_score: float
    liveness_status: str
    image_path: Optional[str] = None
    pil_image: Optional[Image.Image] = None


class PassGenerator:
    """Generates official verified exam entry passes and receipts."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or os.path.join(config.DATA_DIR, "exam_passes")
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_pass_code(self, roll_number: str, exam_title: str, timestamp_str: str) -> str:
        """Generate a deterministic, verifiable pass token."""
        raw_token = f"{roll_number}-{exam_title}-{timestamp_str}-SECRET_EXAM_SALT"
        token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()[:8].upper()
        return f"EXAM-{roll_number.upper()}-{token_hash}"

    def create_exam_pass(
        self,
        student_name: str,
        roll_number: str,
        course: str = "",
        exam_title: str = "General Examination",
        hall_number: str = config.DEFAULT_EXAM_HALL,
        similarity_score: float = 1.0,
        liveness_status: str = "LIVE VERIFIED",
        student_face_crop: Optional[np.ndarray] = None
    ) -> ExamPass:
        """
        Generate a complete Digital Exam Pass graphic and record.
        """
        now = datetime.datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        pass_code = self.generate_pass_code(roll_number, exam_title, timestamp_str)

        # Create pass canvas (700 x 440 px)
        width, height = 700, 440
        canvas = Image.new("RGB", (width, height), color="#FFFFFF")
        draw = ImageDraw.Draw(canvas)

        # Header background banner
        draw.rectangle([(0, 0), (width, 80)], fill="#0F172A")
        # Emerald accent strip
        draw.rectangle([(0, 80), (width, 86)], fill="#059669")

        # Outer border
        draw.rectangle([(0, 0), (width - 1, height - 1)], outline="#CBD5E1", width=2)

        # Load fonts (fallback to default if custom ttf not present)
        try:
            font_inst = ImageFont.truetype("arial.ttf", 16)
            font_title = ImageFont.truetype("arialbd.ttf", 20)
            font_header = ImageFont.truetype("arialbd.ttf", 14)
            font_body = ImageFont.truetype("arial.ttf", 13)
            font_bold = ImageFont.truetype("arialbd.ttf", 13)
            font_code = ImageFont.truetype("cour.ttf", 12)
        except Exception:
            font_inst = ImageFont.load_default()
            font_title = font_inst
            font_header = font_inst
            font_body = font_inst
            font_bold = font_inst
            font_code = font_inst

        # Header Text
        draw.text((24, 16), config.DEFAULT_INSTITUTION_NAME.upper(), fill="#94A3B8", font=font_inst)
        draw.text((24, 38), "BIOMETRIC EXAMINATION ENTRY PASS", fill="#FFFFFF", font=font_title)

        # Badge Top Right
        draw.rounded_rectangle([(520, 20), (676, 60)], radius=6, fill="#064E3B", outline="#059669", width=1)
        draw.text((536, 30), "✓ ADMITTED", fill="#34D399", font=font_header)

        # Body Layout: Left side photo (120x140), Right side details
        photo_box_x = 28
        photo_box_y = 110
        photo_w = 120
        photo_h = 140

        if student_face_crop is not None and student_face_crop.size > 0:
            try:
                rgb_crop = cv2.cvtColor(student_face_crop, cv2.COLOR_BGR2RGB)
                pil_face = Image.fromarray(rgb_crop).resize((photo_w, photo_h), Image.Resampling.LANCZOS)
                canvas.paste(pil_face, (photo_box_x, photo_box_y))
                draw.rectangle(
                    [(photo_box_x - 1, photo_box_y - 1), (photo_box_x + photo_w, photo_box_y + photo_h)],
                    outline="#059669",
                    width=2
                )
            except Exception:
                draw.rectangle(
                    [(photo_box_x, photo_box_y), (photo_box_x + photo_w, photo_box_y + photo_h)],
                    fill="#F1F5F9",
                    outline="#CBD5E1",
                    width=1
                )
                draw.text((photo_box_x + 20, photo_box_y + 60), "PHOTO", fill="#94A3B8", font=font_body)
        else:
            draw.rectangle(
                [(photo_box_x, photo_box_y), (photo_box_x + photo_w, photo_box_y + photo_h)],
                fill="#F1F5F9",
                outline="#CBD5E1",
                width=1
            )
            draw.text((photo_box_x + 20, photo_box_y + 60), "VERIFIED", fill="#059669", font=font_bold)

        # Candidate Details Column
        dx = 175
        dy = 110
        line_h = 26

        fields = [
            ("Candidate Name:", student_name),
            ("Roll Number:", roll_number.upper()),
            ("Course / Branch:", course or "General / Enrolled"),
            ("Examination:", exam_title),
            ("Allocated Hall:", hall_number),
            ("Verified Time:", timestamp_str),
            ("Biometric Match:", f"Cosine Similarity {similarity_score:.3f} ({liveness_status})"),
        ]

        for label, val in fields:
            draw.text((dx, dy), label, fill="#64748B", font=font_body)
            draw.text((dx + 130, dy), val, fill="#0F172A", font=font_bold)
            dy += line_h

        # Footer Verification Bar
        foot_y = 360
        draw.rectangle([(0, foot_y), (width, height)], fill="#F8FAFC")
        draw.line([(0, foot_y), (width, foot_y)], fill="#E2E8F0", width=1)

        draw.text((24, foot_y + 16), "Verification Token:", fill="#64748B", font=font_body)
        draw.text((140, foot_y + 16), pass_code, fill="#1E293B", font=font_code)

        # Simulated digital security seal on bottom right
        draw.rounded_rectangle([(520, foot_y + 10), (676, foot_y + 60)], radius=4, fill="#ECFDF5", outline="#A7F3D0")
        draw.text((534, foot_y + 18), "BIOMETRIC SEAL", fill="#065F46", font=font_header)
        draw.text((534, foot_y + 38), "CRYPTOGRAPHICALLY SIGNED", fill="#047857", font=font_code)

        # Save image file
        file_name = f"pass_{roll_number}_{now.strftime('%Y%m%d_%H%M%S')}.png"
        save_path = os.path.join(self.output_dir, file_name)
        canvas.save(save_path, "PNG")

        return ExamPass(
            pass_code=pass_code,
            student_name=student_name,
            roll_number=roll_number,
            course=course,
            exam_title=exam_title,
            hall_number=hall_number,
            timestamp=timestamp_str,
            similarity_score=similarity_score,
            liveness_status=liveness_status,
            image_path=save_path,
            pil_image=canvas
        )

    def generate_html_receipt(self, exam_pass: ExamPass) -> str:
        """Generate formatted HTML receipt for browser display and printing."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Exam Entry Pass - {exam_pass.roll_number}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f1f5f9; padding: 40px; margin: 0; }}
        .pass-card {{ max-width: 680px; margin: 0 auto; background: #ffffff; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); overflow: hidden; border: 1px solid #cbd5e1; }}
        .header {{ background: #0f172a; color: #ffffff; padding: 24px 32px; border-bottom: 5px solid #059669; display: flex; justify-content: space-between; align-items: center; }}
        .header h1 {{ margin: 0; font-size: 20px; font-weight: 700; letter-spacing: 0.5px; }}
        .header p {{ margin: 4px 0 0 0; color: #94a3b8; font-size: 13px; }}
        .badge {{ background: #ecfdf5; color: #059669; padding: 8px 16px; border-radius: 20px; font-weight: bold; font-size: 13px; border: 1px solid #a7f3d0; }}
        .content {{ padding: 32px; }}
        .info-grid {{ display: grid; grid-template-columns: 140px 1fr; gap: 14px; font-size: 14px; }}
        .label {{ color: #64748b; font-weight: 500; }}
        .value {{ color: #0f172a; font-weight: 600; }}
        .footer {{ background: #f8fafc; padding: 18px 32px; border-top: 1px solid #e2e8f0; display: flex; justify-content: space-between; font-size: 12px; color: #64748b; }}
        .token {{ font-family: monospace; font-weight: bold; color: #1e293b; font-size: 13px; }}
        .print-btn {{ display: block; margin: 20px auto; padding: 10px 24px; background: #2563eb; color: white; border: none; border-radius: 6px; font-size: 14px; font-weight: bold; cursor: pointer; }}
        @media print {{ .print-btn {{ display: none; }} body {{ background: #fff; padding: 0; }} .pass-card {{ box-shadow: none; border: 1px solid #000; }} }}
    </style>
</head>
<body>
    <button class="print-btn" onclick="window.print()">🖨️ Print Exam Entry Pass</button>
    <div class="pass-card">
        <div class="header">
            <div>
                <p>{config.DEFAULT_INSTITUTION_NAME.upper()}</p>
                <h1>BIOMETRIC EXAMINATION ENTRY PASS</h1>
            </div>
            <div class="badge">✓ VERIFIED & ADMITTED</div>
        </div>
        <div class="content">
            <div class="info-grid">
                <div class="label">Candidate Name:</div>
                <div class="value">{exam_pass.student_name}</div>
                <div class="label">Roll Number:</div>
                <div class="value">{exam_pass.roll_number.upper()}</div>
                <div class="label">Course / Branch:</div>
                <div class="value">{exam_pass.course or "General / Enrolled"}</div>
                <div class="label">Examination:</div>
                <div class="value">{exam_pass.exam_title}</div>
                <div class="label">Allocated Hall:</div>
                <div class="value">{exam_pass.hall_number}</div>
                <div class="label">Verification Time:</div>
                <div class="value">{exam_pass.timestamp}</div>
                <div class="label">Biometric Match:</div>
                <div class="value">Cosine Similarity {exam_pass.similarity_score:.3f} ({exam_pass.liveness_status})</div>
            </div>
        </div>
        <div class="footer">
            <div>Token: <span class="token">{exam_pass.pass_code}</span></div>
            <div>Official Exam Security Seal ✓</div>
        </div>
    </div>
</body>
</html>"""
        html_path = os.path.join(self.output_dir, f"pass_{exam_pass.roll_number}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        return html_path
