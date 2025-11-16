import logging
from typing import Dict

logger = logging.getLogger(__name__)


def extract_metadata_from_chunk(chunk_text: str, chunk_index: int) -> Dict[str, any]:
    text_lower = chunk_text.lower()
    metadata = {}
    
    service_patterns = {
        "payroll": ["payroll", "pay stub", "paycheck", "salary", "wage", "compensation", "direct deposit", "w-2", "tax"],
        "benefits": ["health insurance", "dental insurance", "vision insurance", "401k", "retirement plan", "hsa", "fsa", "benefits enrollment"],
        "time-attendance": ["timesheet", "time tracking", "clock in", "clock out", "attendance", "time off request", "pto request"],
        "employee-portal": ["employee portal", "self-service portal", "employee self-service", "my portal", "employee dashboard"],
        "hr-support": ["hr department", "human resources", "hr contact", "hr help", "hr assistance"],
        "expense": ["expense report", "reimbursement", "travel expense", "expense claim", "mileage reimbursement"],
    }
    
    metadata["service_name"] = "general"
    for service, patterns in service_patterns.items():
        if any(pattern in text_lower for pattern in patterns):
            metadata["service_name"] = service
            break
    
    if any(term in text_lower for term in ["troubleshooting", "support", "help", "issue", "problem", "error", "fix", "resolve"]):
        metadata["section"] = "troubleshooting"
    elif any(term in text_lower for term in ["leave", "vacation", "sick leave", "pto", "fmla", "personal time", "holiday"]):
        metadata["section"] = "leave-policies"
    elif any(term in text_lower for term in ["benefits", "insurance", "401k", "retirement", "health plan", "dental plan", "vision plan"]):
        metadata["section"] = "benefits"
    elif any(term in text_lower for term in ["payroll", "pay", "salary", "wage", "compensation", "paycheck", "direct deposit"]):
        metadata["section"] = "payroll"
    elif any(term in text_lower for term in ["policy", "policies", "code of conduct", "harassment", "discrimination", "workplace policy"]):
        metadata["section"] = "policies"
    elif any(term in text_lower for term in ["procedure", "process", "how to", "workflow", "steps", "guide", "instructions"]):
        metadata["section"] = "procedures"
    elif any(term in text_lower for term in ["performance", "review", "goal", "development", "career", "appraisal"]):
        metadata["section"] = "performance"
    elif any(term in text_lower for term in ["compliance", "legal", "fmla", "ada", "eeoc", "workers comp", "labor law"]):
        metadata["section"] = "compliance"
    else:
        metadata["section"] = "general"
    
    if any(term in text_lower for term in ["error", "troubleshoot", "support", "help", "issue", "problem"]):
        metadata["doc_type"] = "support_kb"
    elif any(term in text_lower for term in ["procedure", "step", "how to", "process", "workflow", "guide"]):
        metadata["doc_type"] = "procedure"
    elif any(term in text_lower for term in ["policy", "policies", "standard", "guideline", "rule"]):
        metadata["doc_type"] = "policy"
    else:
        metadata["doc_type"] = "hr_documentation"
    
    tags = []
    hr_keywords = ["leave", "vacation", "benefits", "payroll", "401k", "insurance", 
                   "performance", "review", "training", "compliance", "policy", "hr",
                   "employee", "portal", "self-service", "expense", "reimbursement",
                   "timesheet", "attendance", "pto", "fmla", "health", "dental", "vision"]
    for keyword in hr_keywords:
        if keyword in text_lower:
            tags.append(keyword)
    
    if metadata.get("section"):
        tags.append(metadata["section"])
    if metadata.get("service_name") and metadata["service_name"] != "general":
        tags.append(metadata["service_name"])
    
    metadata["tags"] = list(set(tags))
    
    return metadata


def parse_query_filters(user_query: str) -> tuple[str, Dict[str, any]]:
    query_lower = user_query.lower()
    filters = {}
    search_terms = user_query
    
    service_keywords = {
        "payroll": "payroll",
        "pay stub": "payroll",
        "paycheck": "payroll",
        "salary": "payroll",
        "direct deposit": "payroll",
        "benefits": "benefits",
        "health insurance": "benefits",
        "401k": "benefits",
        "retirement": "benefits",
        "timesheet": "time-attendance",
        "time tracking": "time-attendance",
        "attendance": "time-attendance",
        "time off": "time-attendance",
        "leave": "time-attendance",
        "vacation": "time-attendance",
        "pto": "time-attendance",
        "employee portal": "employee-portal",
        "self-service": "employee-portal",
        "portal": "employee-portal",
        "expense": "expense",
        "reimbursement": "expense",
        "expense report": "expense",
        "hr": "hr-support",
        "human resources": "hr-support",
    }
    
    for keyword, service in service_keywords.items():
        if keyword in query_lower:
            filters["service_name"] = service
            break
    
    if any(term in query_lower for term in ["troubleshoot", "support", "help", "error", "issue", "problem", "fix"]):
        filters["section"] = "troubleshooting"
    elif any(term in query_lower for term in ["leave", "vacation", "sick", "time off", "pto", "fmla", "holiday"]):
        filters["section"] = "leave-policies"
    elif any(term in query_lower for term in ["benefits", "insurance", "401k", "retirement", "health", "dental", "vision"]):
        filters["section"] = "benefits"
    elif any(term in query_lower for term in ["payroll", "pay", "salary", "paycheck", "wage", "direct deposit"]):
        filters["section"] = "payroll"
    elif any(term in query_lower for term in ["policy", "policies", "code of conduct", "workplace policy"]):
        filters["section"] = "policies"
    elif any(term in query_lower for term in ["procedure", "process", "how to", "workflow", "steps", "guide"]):
        filters["section"] = "procedures"
    elif any(term in query_lower for term in ["performance", "review", "goal", "appraisal"]):
        filters["section"] = "performance"
    elif any(term in query_lower for term in ["compliance", "legal", "fmla", "ada", "eeoc"]):
        filters["section"] = "compliance"
    
    return search_terms, filters
