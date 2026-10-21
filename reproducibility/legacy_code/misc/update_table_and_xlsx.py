#!/usr/bin/env python3
"""Regenerate TABLE_source_final.svg (real text) WITH a footnote flagging no-activity sources,
and update the Excel workbook with corrected patent counts + notes + activity-vs-coverage sheet.
"""
import warnings; warnings.filterwarnings('ignore')
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
RDLogger=None
BASE="/storage/savi/saveenas/Projects/SynGlue"; REV=f"{BASE}/revision"
OUT=f"{REV}/rebuttal_analyses/17_source_to_final"; os.makedirs(OUT, exist_ok=True)
XLS=f"{REV}/rebuttal_analyses/16_final_coverage_viz/outputs/DATASET_ACCOUNTABILITY.xlsx"
for n,p in {"NimbusSans-Regular":"/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",
            "NimbusSans-Bold":"/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf"}.items():
    if os.path.exists(p):
        try: font_manager.fontManager.addfont(p)
        except: pass
plt.rcParams.update({'font.family':'Arial','svg.fonttype':'none'})

rows=[
 ["PROTAC-DB (3.0)","9,380","6,110","5,698","441","21","527","ACTIVITY (training)"],
 ["TACK","6,561","3,514","2,723","109","9","21","ACTIVITY (DC50/Dmax)"],
 ["TPDdb","4,293","2,426","1,753","23","2","21","ACTIVITY (DC50/Dmax)"],
 ["PROTACpedia","871","585","529","60","7","0","ACTIVITY (DC50/Dmax)"],
 ["PROTAC-Patent (full)","63,136","57,533","–","223","0*","0*","NO activity (coverage only)"],
 ["External activity set (novel-to-DB3)","–","2,562","–","–","–","21","ACTIVITY (test)"],
 ["External coverage union (incl patent)","–","≈59,000","–","223","–","–","COVERAGE only"],
]
cols=["Source","Records","Unique","Scaffolds","Targets","E3","Assay","Has activity endpoint?"]
fig,ax=plt.subplots(figsize=(11,3.6)); ax.axis('off')
tab=ax.table(cellText=[cols]+rows,loc='center',cellLoc='center')
tab.auto_set_font_size(False); tab.set_fontsize(8); tab.scale(0.95,1.6)
for (r,c),cell in tab.get_celld().items():
    cell.set_edgecolor('white'); cell.set_linewidth(1.2)
    if r==0: cell.set_facecolor('#4C72B0'); cell.set_text_props(color='white',weight='bold')
    elif r%2==0: cell.set_facecolor('#EEF1F7')
    if c==0 and r>0: cell.set_text_props(weight='bold')
    if r>0 and 'NO activity' in str(cell.get_text()): cell.get_text().set_color('#C44E52'); cell.get_text().set_weight('bold')
# footnote
foot=("* PROTAC-PatentDB & PROTACpedia carry NO degradation activity endpoint "
      "(no DC50/Dmax, no active/inactive label) and no E3 label.\n"
      "  They are used for coverage / scaffold / target-diversity / ADMET analyses only, NOT for activity scoring.")
ax.text(0.5,-0.06,foot,transform=ax.transAxes,ha='center',va='top',fontsize=7,color='#333')
fig.tight_layout()
fig.savefig(f"{OUT}/TABLE_source_final.svg"); fig.savefig(f"{OUT}/TABLE_source_final.png"); plt.close(fig)
print("saved TABLE_source_final with footnote")

# ── update Excel workbook ──
import openpyxl
wb=openpyxl.load_workbook(XLS)
# 1) fix Dataset_Counts PROTAC-Patent row (63,136 / 57,533 / targets 223)
ws=wb['Dataset_Counts']
pro=0
for r in range(2, ws.max_row+1):
    if ws.cell(r,1).value and 'PROTAC-Patent' in str(ws.cell(r,1).value):
        ws.cell(r,2).value="63,136"; ws.cell(r,3).value="57,533"; ws.cell(r,5).value="223"; ws.cell(r,6).value="0 (n/a)"
        ws.cell(r,1).value="PROTAC-PatentDB (full, SciData 2025)"
        pro=r
# 2) add Has_activity column to Dataset_Counts
hdr_row=1
col=ws.max_column+1
ws.cell(hdr_row,col).value="Has activity endpoint?"
act_map={"PROTAC-DB": "ACTIVITY (training)","Model":"ACTIVITY","TACK":"ACTIVITY (DC50/Dmax)",
 "TPDdb":"ACTIVITY (DC50/Dmax)","PROTACpedia":"ACTIVITY (DC50/Dmax)","PROTAC-PatentDB":"NO (coverage only)",
 "EXTERNAL POOL":"ACTIVITY (test)","final_dataset":"ACTIVITY","External not-in-training":"ACTIVITY (test)"}
for r in range(2, ws.max_row+1):
    nm=str(ws.cell(r,1).value or "")
    ws.cell(r,col).value = "NO (coverage/ADMET only)" if "PatentDB" in nm else \
        ("ACTIVITY (DC50/Dmax)" if "TACK" in nm or "External" in nm else \
         ("ACTIVITY (training)" if "PROTAC-DB" in nm and "test" not in nm else "ACTIVITY"))
# 3) new sheet: activity vs coverage
if "External_Activity_vs_Coverage" in wb.sheetnames: del wb["External_Activity_vs_Coverage"]
ws2=wb.create_sheet("External_Activity_vs_Coverage")
info=[["Set","Size","Has activity (DC50/Dmax/E3)?","Use"],
 ["TACK-novel (activity test)","2,562","YES","model ranking, AUROC, EF10%, rank correlation"],
 ["External pool (no PROTAC-DB)","2,930","YES","activity benchmark (exact-only hold-out)"],
 ["Final external union (incl full patent)","≈59,000","NO (patent lacks endpoint)","coverage, scaffolds, target diversity, ADMET"],
 ["PROTAC-PatentDB (full)","57,533","NO","coverage/diversity/ADMET only"],
 ["PROTACpedia","585","PARTIAL (74% are PROTAC-DB dups)","mostly leak; little novelty"]]
for i,row in enumerate(info, start=1):
    for j,val in enumerate(row, start=1):
        ws2.cell(i,j).value=val
for j in range(1,5):
    ws2.cell(1,j).font=openpyxl.styles.Font(bold=True,color='FFFFFF')
    ws2.cell(1,j).fill=openpyxl.styles.PatternFill('solid',fgColor='4C72B0')
# 4) fix PROTACDB_published note? leave as is.
wb.save(XLS)
print("updated Excel:", XLS)
