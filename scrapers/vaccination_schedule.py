import pdfplumber
import re
import json

def vaccination_pdf_to_json(pdf_path, json_out="vaccination_schedule.json"):
    data = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            lines = text.split("\n")

            stage = None
            for line in lines:
                line = line.strip()

                if re.search(r"Pregnant Women|Infants|Children", line, re.IGNORECASE):
                    stage = line
                    continue

                match = re.match(r"(.+?):\s*(.+)", line)
                if match and stage:
                    time = match.group(1).strip()
                    vaccines = [v.strip() for v in match.group(2).split(",")]
                    for vaccine in vaccines:
                        data.append({
                            "stage": stage,
                            "time": time,
                            "vaccine": vaccine
                        })

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    return data


if __name__ == "__main__":
    pdf_path = r"/Users/aashutoshkumar/Documents/Projects/healthgraph-assistant/data/National_ Immunization_Schedule.pdf"
    schedule_json = vaccination_pdf_to_json(pdf_path)
    print(schedule_json)