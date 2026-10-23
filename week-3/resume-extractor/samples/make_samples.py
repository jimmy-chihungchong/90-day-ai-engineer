"""Generate 10 synthetic test resumes (PDF + DOCX) and expected.json (ground truth).

Run:  python samples/make_samples.py
Years are the hand-computed truth (calendar months, overlaps merged); "today" for current jobs
is the date the script runs.
"""
from __future__ import annotations

import io
import json
from datetime import date
from pathlib import Path

from docx import Document
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.pdfgen import canvas

OUT = Path(__file__).parent / "resumes"
OUT.mkdir(exist_ok=True)
TODAY = date.today()
EXPECTED: dict[str, dict] = {}


def months_to_today(y: int, m: int) -> int:
    return (TODAY.year - y) * 12 + (TODAY.month - m) + 1


def years(months: int) -> float:
    return round(months / 12, 1)


def expect(file, name, email, yrs, skills, jobs, tol=0.6, note=""):
    EXPECTED[file] = dict(name=name, email=email, years=yrs, skills=skills, jobs=jobs, tol=tol, note=note)


def pdf(path, lines, size=10, lead=14):
    c = canvas.Canvas(str(path), pagesize=letter)
    y = 750
    for ln in lines:
        for part in (simpleSplit(ln, "Helvetica", size, 500) or [""]):
            if y < 50:
                c.showPage(); y = 750
            c.setFont("Helvetica", size); c.drawString(56, y, part); y -= lead
    c.save()


def docx(path, lines, header=None):
    d = Document()
    if header:
        d.sections[0].header.paragraphs[0].text = header
    for ln in lines:
        d.add_paragraph(ln)
    d.save(str(path))


# 01 + 02: baselines -------------------------------------------------------------------------
L1 = ["Alex Morgan", "alex.morgan@example.com | 555-0101 | Austin, TX", "", "EXPERIENCE",
      "Data Engineer, Initech            Feb 2021 - Jun 2024", "- Built Airflow pipelines on AWS",
      "Junior Analyst, Globex            Jan 2018 - Dec 2020", "- Wrote SQL reports in Python",
      "", "SKILLS", "Python, SQL, Docker, AWS, Airflow", "", "EDUCATION", "B.Sc. Statistics, UT Austin, 2017"]
docx(OUT / "01_simple.docx", L1)
expect("01_simple.docx", "Alex Morgan", "alex.morgan@example.com", 6.4,
       ["Python", "SQL", "Docker", "AWS"], 2, note="baseline DOCX; 2021-01 gap")

L2 = ["Priya Natarajan", "priya.n@example.com | +44 20 7946 0000 | London, UK", "", "PROFESSIONAL EXPERIENCE",
      "Senior Backend Developer - Fintrust    09/2019 - 12/2023", "* Designed Kafka event pipeline in Java",
      "Backend Developer - Paynet    09/2016 - 08/2019", "* Spring Boot microservices on PostgreSQL",
      "", "TECHNICAL SKILLS", "Java, Spring Boot, Kafka, PostgreSQL, Kubernetes", "", "EDUCATION",
      "MSc Computer Science, Imperial College London, 2016"]
pdf(OUT / "02_simple.pdf", L2)
expect("02_simple.pdf", "Priya Natarajan", "priya.n@example.com", 7.3,
       ["Java", "Spring Boot", "Kafka", "PostgreSQL"], 2, note="baseline PDF; numeric dates")

# 03: two-column PDF (sidebar + main, same baselines so text interleaves) -----------------
left = ["Daniel Okafor", "daniel.o@example.com", "555-0103", "Berlin, Germany", "", "SKILLS", "Go", "Kubernetes",
        "Terraform", "Prometheus", "Linux", "", "EDUCATION", "B.Eng. Electronics", "Univ. of Lagos, 2014"]
right = ["EXPERIENCE", "", "Platform Engineer - Cloudly", "Jun 2018 - Nov 2022", "Ran 40 Kubernetes clusters",
         "and built Terraform modules.", "", "DevOps Engineer - Netwave", "Mar 2015 - May 2018",
         "Automated deploys with Go tooling", "and monitored with Prometheus."]
c = canvas.Canvas(str(OUT / "03_two_column.pdf"), pagesize=letter)
y = 740
for i in range(max(len(left), len(right))):
    c.setFont("Helvetica", 10)
    if i < len(left): c.drawString(40, y, left[i])
    if i < len(right): c.drawString(230, y, right[i])
    y -= 16
c.save()
expect("03_two_column.pdf", "Daniel Okafor", "daniel.o@example.com", 7.8,
       ["Go", "Kubernetes", "Terraform", "Prometheus"], 2, note="two-column PDF, lines interleave")

# 04: tables + header name -----------------------------------------------------------------
d = Document()
d.sections[0].header.paragraphs[0].text = "Sofia Rossi  |  sofia.rossi@example.com  |  Milan, Italy"
d.add_paragraph("Frontend Engineer")
t = d.add_table(rows=3, cols=3)
rows = [("Dates", "Role", "Details"),
        ("07/2021 - 03/2025", "Senior Frontend Engineer, Modena Labs", "Led React/TypeScript design system"),
        ("01/2019 - 06/2021", "Frontend Developer, Pixel Forge", "Built GraphQL clients, Node.js BFF")]
for r, row in enumerate(rows):
    for ci, v in enumerate(row):
        t.cell(r, ci).text = v
t2 = d.add_table(rows=2, cols=2)
t2.cell(0, 0).text, t2.cell(0, 1).text = "Skills", "React, TypeScript, Node.js, GraphQL, Jest"
t2.cell(1, 0).text, t2.cell(1, 1).text = "Languages", "Italian (native), English (C1)"
d.save(str(OUT / "04_tables_header.docx"))
expect("04_tables_header.docx", "Sofia Rossi", "sofia.rossi@example.com", 6.3,
       ["React", "TypeScript", "Node.js", "GraphQL"], 2, note="name only in header; data in tables")

# 05: scanned (image-only) PDF -------------------------------------------------------------
img = Image.new("RGB", (1240, 800), "white")
dr = ImageDraw.Draw(img)
font = ImageFont.load_default(size=28)
for i, ln in enumerate(["Liam Walker", "liam.walker@example.com", "EXPERIENCE",
                        "Electrician, Brightwire  2012 - 2024", "SKILLS", "Wiring, Safety audits, AutoCAD"]):
    dr.text((60, 60 + i * 60), ln, fill="black", font=font)
buf = io.BytesIO(); img.save(buf, "PNG"); buf.seek(0)
c = canvas.Canvas(str(OUT / "05_scanned.pdf"), pagesize=letter)
c.drawImage(ImageReader(buf), 20, 300, width=570, height=368); c.save()
expect("05_scanned.pdf", "Liam Walker", "liam.walker@example.com", None, [], 0,
       note="image-only PDF; correct behaviour is a clear 'scanned' error", )
EXPECTED["05_scanned.pdf"]["expect_error"] = True

# 06: Spanish ------------------------------------------------------------------------------
L6 = ["Carmen Ruiz Salazar", "carmen.ruiz@example.com | +34 600 000 000 | Madrid, España", "",
      "EXPERIENCIA PROFESIONAL", "Analista Senior de Datos - Consultora Sol    abril 2022 - actualidad",
      "- Dashboards en Power BI y modelos en Python", "Ingeniera de Datos - Banco Norte    enero 2018 - marzo 2022",
      "- Pipelines con Spark y SQL", "", "HABILIDADES", "Python, Spark, Power BI, SQL, Azure", "",
      "FORMACIÓN", "Grado en Matemáticas, Universidad Complutense, 2017"]
docx(OUT / "06_spanish.docx", L6)
expect("06_spanish.docx", "Carmen Ruiz Salazar", "carmen.ruiz@example.com",
       years(months_to_today(2018, 1)), ["Python", "Spark", "Power BI", "SQL"], 2,
       note="Spanish month names and 'actualidad' (= present)")

# 07: very long PDF (truncation); skills section sits at the very end ------------------------
lines = ["Robert Chen", "robert.chen@example.com | Seattle, WA", "", "EXPERIENCE"]
bullet = ("Delivered cross-functional initiatives spanning architecture reviews, capacity planning, incident "
          "response, mentoring, vendor evaluation, and quarterly roadmap alignment with product leadership.")
for i, yr in enumerate(range(2024, 2000, -2)):
    lines += ["", f"Engineering Manager {i + 1}, Company {i + 1}      Jan {yr} - Dec {yr + 1}"]
    lines += [f"- {bullet} (item {k})" for k in range(1, 15)]
lines += ["", "TECHNICAL SKILLS", "Rust, C++, Distributed Systems, Kubernetes, Leadership"]
pdf(OUT / "07_long.pdf", lines, size=9, lead=11)
expect("07_long.pdf", "Robert Chen", "robert.chen@example.com", 24.0,
       ["Rust", "C++", "Kubernetes"], 12, note="~35k chars; text is truncated at 24k so oldest jobs + skills are lost")

# 08: messy dates (overlap, gap, season, year-only) -----------------------------------------
L8 = ["Nadia Haddad", "nadia.haddad@example.com | Toronto, ON", "", "WORK",
      "UX Designer, Studio Q          03/2018 - 08/2021", "Part-time Design Teacher, City College   09/2019 - 06/2021",
      "Design Intern, Pixel Co        Summer 2017", "Freelance Illustrator, self-employed    2014 - 2016", "",
      "SKILLS", "Figma, Sketch, Illustration, User Research"]
docx(OUT / "08_messy_dates.docx", L8)
expect("08_messy_dates.docx", "Nadia Haddad", "nadia.haddad@example.com", 6.8,
       ["Figma", "Sketch", "User Research"], 4, tol=0.5,
       note="overlap, gap, 'Summer 2017' (~3 months, one-sided), year-only freelance")

# 09: creative layout, short year forms, skills in prose ------------------------------------
L9 = ["JORDAN BLAKE", "hello@jordanblake.example.com  ~  jordanblake.example.com  ~  Denver", "",
      "Where I've been",
      "Growth Lead @ Lumen  ('21 - now)", "~ Doubled signups with lifecycle email experiments",
      "Marketing Associate @ BrightPath  ('17 - '21)", "~ Ran paid social and content", "",
      "What I'm good at",
      "I'm fluent in SEO, Google Analytics and HubSpot, and I can get by in SQL.", "",
      "Schooling", "BA Communications, CU Boulder, class of 2017"]
pdf(OUT / "09_creative.pdf", L9)
expect("09_creative.pdf", "Jordan Blake", "hello@jordanblake.example.com", 9.8,
       ["SEO", "Google Analytics", "HubSpot"], 2, tol=1.2,
       note="'21 - now style dates; skills written as a sentence; year-only ranges")

# 10: student, no work experience ---------------------------------------------------------
L10 = ["Mei Tanaka", "mei.tanaka@example.com | Osaka, Japan", "", "EDUCATION",
       "B.S. Data Science (expected 2027), Osaka University", "", "PROJECTS",
       "House price predictor - Python, pandas, scikit-learn (2025)",
       "Kaggle Titanic top 10% - gradient boosting (2024)", "", "SKILLS", "Python, pandas, scikit-learn, Git"]
docx(OUT / "10_student.docx", L10)
expect("10_student.docx", "Mei Tanaka", "mei.tanaka@example.com", None,
       ["Python", "pandas", "scikit-learn"], 0, note="no jobs: years must be null; projects are not jobs")

(OUT.parent / "expected.json").write_text(json.dumps(EXPECTED, indent=2), encoding="utf-8")
print(f"wrote {len(EXPECTED)} resumes to {OUT}")
