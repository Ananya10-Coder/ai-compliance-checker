import json 
import os 
from dotenv import load_dotenv
from typing import Dict, Any
from langchain_groq import ChatGroq
from .reporter import generate_report

load_dotenv()
def rewrite_document(file_path: str) -> str:
    """
    Takes an input file, runs compliance analysis,
    and rewrites the document to be compliant with policies.
    Includes rewritten text, fix summary, and citations table.
    """
    report_data = generate_report(file_path)
    original_chunks = report_data.get("original_chunks", [])
    structured_issues = report_data.get("issues", [])
    raw_llm_output = report_data.get("raw_llm_output", "")
    
    original_text = "\n".join([chunk['text'] for chunk in original_chunks])

    issues_summary = ""
    for idx, issue in enumerate(structured_issues, start=1):
        issues_summary += f"Issue {idx}:\n"
        issues_summary += f"- Citation: {issue.get('citation','N/A')}\n"
        issues_summary += f"- Violated Policy: {issue.get('violated_policy','N/A')}\n"
        issues_summary += f"- Reasoning: {issue.get('reasoning','N/A')}\n"
        issues_summary += f"- Problematic Snippet: {issue.get('snippet','N/A')}\n"
        issues_summary += f"- Improvement: {issue.get('improvement','N/A')}\n\n"

    citations_table = "| Citation | Problematic Snippet | Policy Violated |\n|----------|----------------------|-----------------|\n"
    for issue in structured_issues:
        citations_table += f"| {issue.get('citation','N/A')} | {issue.get('snippet','N/A')} | {issue.get('violated_policy','N/A')} |\n"

    llm = ChatGroq(
        api_key= os.getenv("GROQ_API_KEY"),
        model = "llama3-8b-8192"
    )

    prompt = f"""
    You are a compliance officer and professional editor.

    You are given:
    1. The original document text:
    {original_text}

    2. Structured compliance report from automated analysis:
    {issues_summary}

    3. Raw output from the automated analysis LLM (may contain extra context):
    {raw_llm_output}

        Task:
    - Rewrite the **original document itself** to make it fully compliant.
    - Preserve the structure, style, and intent of the original text.
    - For each problematic snippet, replace it with a compliant version that maintains clarity and professionalism.
    - Do NOT delete text without replacing it — always substitute with compliant language.
    - Ensure the rewritten version flows naturally and looks production-ready.

    Output in the following format (Markdown):

    === REWRITTEN DOCUMENT ===
    <rewritten compliant document>

    === FIX SUMMARY ===
    For each issue:
    - Citation: <chunk/section reference>
    - Violated Policy: "<policy>"
    - Problematic Snippet: "<original text that violated>"
    - Fix Applied: "<replacement + reasoning>"

    === CITATIONS & REFERENCES ===
    {citations_table}
    """
    response = llm.invoke(prompt)
    return response.content

if __name__ == "__main__":
    file_path = "data/projects/test_report.txt" 
    rewritten_document = rewrite_document(file_path)
    print(rewritten_document)

