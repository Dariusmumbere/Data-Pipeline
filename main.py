import pandas as pd
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClaimProcessor:
    """Process claims from multiple EMR sources and determine resubmission eligibility"""
    
    # Known retryable and non-retryable denial reasons
    RETRYABLE_REASONS = {"missing modifier", "incorrect npi", "prior auth required"}
    NON_RETRYABLE_REASONS = {"authorization expired", "incorrect provider type"}
    
    def __init__(self, current_date: str = "2025-07-30"):
        self.current_date = datetime.strptime(current_date, "%Y-%m-%d")
        self.processed_count = 0
        self.resubmission_candidates = []
        self.rejected_records = []
        self.source_counts = {"alpha": 0, "beta": 0}
        
    def normalize_date(self, date_str: str) -> Optional[str]:
        """Normalize various date formats to ISO format"""
        if not date_str or pd.isna(date_str):
            return None
            
        try:
            # Handle different date formats
            if "T" in date_str:
                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            else:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
            return dt.isoformat()
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse date: {date_str}, error: {e}")
            return None
    
    def normalize_reason(self, reason: Any) -> Optional[str]:
        """Normalize denial reason text"""
        if not reason or pd.isna(reason) or reason == "None":
            return None
            
        # Convert to lowercase and strip whitespace
        normalized = str(reason).lower().strip()
        return normalized if normalized else None
    
    def classify_denial_reason(self, reason: Optional[str]) -> str:
        """Classify denial reason as retryable, non-retryable, or ambiguous"""
        if not reason:
            return "ambiguous"
            
        normalized_reason = reason.lower()
        
        if normalized_reason in self.RETRYABLE_REASONS:
            return "retryable"
        elif normalized_reason in self.NON_RETRYABLE_REASONS:
            return "non_retryable"
        else:
            return "ambiguous"
    
    def process_alpha_source(self, file_path: str) -> List[Dict]:
        """Process EMR Alpha CSV data"""
        try:
            df = pd.read_csv(file_path)
            records = []
            
            for _, row in df.iterrows():
                try:
                    normalized_record = {
                        "claim_id": str(row["claim_id"]) if pd.notna(row["claim_id"]) else None,
                        "patient_id": str(row["patient_id"]) if pd.notna(row["patient_id"]) else None,
                        "procedure_code": str(row["procedure_code"]) if pd.notna(row["procedure_code"]) else None,
                        "denial_reason": self.normalize_reason(row["denial_reason"]),
                        "status": str(row["status"]).lower() if pd.notna(row["status"]) else None,
                        "submitted_at": self.normalize_date(row["submitted_at"]),
                        "source_system": "alpha"
                    }
                    records.append(normalized_record)
                    self.source_counts["alpha"] += 1
                except Exception as e:
                    self.rejected_records.append({
                        "raw_data": row.to_dict(),
                        "error": f"Failed to process alpha record: {e}"
                    })
                    logger.error(f"Error processing alpha record: {row.to_dict()}, error: {e}")
            
            return records
        except Exception as e:
            logger.error(f"Failed to read alpha source: {e}")
            return []
    
    def process_beta_source(self, file_path: str) -> List[Dict]:
        """Process EMR Beta JSON data"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            records = []
            
            for item in data:
                try:
                    normalized_record = {
                        "claim_id": str(item["id"]) if item.get("id") else None,
                        "patient_id": str(item["member"]) if item.get("member") else None,
                        "procedure_code": str(item["code"]) if item.get("code") else None,
                        "denial_reason": self.normalize_reason(item.get("error_msg")),
                        "status": str(item["status"]).lower() if item.get("status") else None,
                        "submitted_at": self.normalize_date(item.get("date")),
                        "source_system": "beta"
                    }
                    records.append(normalized_record)
                    self.source_counts["beta"] += 1
                except Exception as e:
                    self.rejected_records.append({
                        "raw_data": item,
                        "error": f"Failed to process beta record: {e}"
                    })
                    logger.error(f"Error processing beta record: {item}, error: {e}")
            
            return records
        except Exception as e:
            logger.error(f"Failed to read beta source: {e}")
            return []
    
    def is_eligible_for_resubmission(self, claim: Dict) -> bool:
        """Determine if a claim is eligible for resubmission"""
        # Check if status is denied
        if claim.get("status") != "denied":
            return False
            
        # Check if patient_id is not null
        if not claim.get("patient_id"):
            return False
            
        # Check if submitted more than 7 days ago
        submitted_at = claim.get("submitted_at")
        if not submitted_at:
            return False
            
        try:
            submitted_date = datetime.fromisoformat(submitted_at.replace("Z", "+00:00"))
            days_diff = (self.current_date - submitted_date).days
            if days_diff <= 7:
                return False
        except (ValueError, TypeError):
            return False
        
        # Check denial reason
        denial_reason = claim.get("denial_reason")
        reason_classification = self.classify_denial_reason(denial_reason)
        
        if reason_classification == "non_retryable":
            return False
        elif reason_classification == "retryable":
            return True
        else:  # ambiguous - use LLM classifier (mocked for this implementation)
            # In a real implementation, this would call an LLM API
            return self.mock_llm_classifier(denial_reason)
    
    def mock_llm_classifier(self, reason: Optional[str]) -> bool:
        """Mock LLM classifier for ambiguous denial reasons"""
        # In a real implementation, this would call an actual LLM API
        # For this exercise, we'll use a simple rule-based approach
        
        if not reason:
            return False
            
        # Simple heuristic: if the reason contains certain keywords, consider it retryable
        retryable_keywords = {"incorrect", "missing", "required", "incomplete"}
        non_retryable_keywords = {"expired", "not covered", "not billable", "invalid"}
        
        reason_lower = reason.lower()
        
        # Check for non-retryable keywords first
        for keyword in non_retryable_keywords:
            if keyword in reason_lower:
                return False
                
        # Check for retryable keywords
        for keyword in retryable_keywords:
            if keyword in reason_lower:
                return True
                
        # Default to not retryable for ambiguous cases
        return False
    
    def generate_resubmission_recommendation(self, claim: Dict) -> Dict:
        """Generate resubmission recommendation based on denial reason"""
        denial_reason = claim.get("denial_reason", "").lower() if claim.get("denial_reason") else ""
        
        if "missing modifier" in denial_reason:
            return {
                "resubmission_reason": "Missing modifier",
                "recommended_changes": "Add appropriate modifier and resubmit"
            }
        elif "incorrect npi" in denial_reason:
            return {
                "resubmission_reason": "Incorrect NPI",
                "recommended_changes": "Review NPI number and resubmit"
            }
        elif "prior auth required" in denial_reason:
            return {
                "resubmission_reason": "Prior auth required",
                "recommended_changes": "Obtain prior authorization and resubmit"
            }
        elif "incorrect procedure" in denial_reason:
            return {
                "resubmission_reason": "Incorrect procedure code",
                "recommended_changes": "Verify procedure code and resubmit with correct code"
            }
        else:
            return {
                "resubmission_reason": "Other resolvable issue",
                "recommended_changes": "Review claim details and resubmit with corrections"
            }
    
    def process_claims(self, alpha_path: str, beta_path: str) -> None:
        """Main method to process claims from both sources"""
        logger.info("Starting claim processing pipeline")
        
        # Process both data sources
        alpha_claims = self.process_alpha_source(alpha_path)
        beta_claims = self.process_beta_source(beta_path)
        
        all_claims = alpha_claims + beta_claims
        self.processed_count = len(all_claims)
        
        logger.info(f"Processed {self.processed_count} total claims")
        logger.info(f"Alpha claims: {self.source_counts['alpha']}")
        logger.info(f"Beta claims: {self.source_counts['beta']}")
        
        # Check eligibility for each claim
        resubmission_count = 0
        ineligible_count = 0
        
        for claim in all_claims:
            if self.is_eligible_for_resubmission(claim):
                resubmission_count += 1
                recommendation = self.generate_resubmission_recommendation(claim)
                
                resubmission_candidate = {
                    "claim_id": claim["claim_id"],
                    "resubmission_reason": recommendation["resubmission_reason"],
                    "source_system": claim["source_system"],
                    "recommended_changes": recommendation["recommended_changes"]
                }
                
                self.resubmission_candidates.append(resubmission_candidate)
            else:
                ineligible_count += 1
        
        logger.info(f"Claims flagged for resubmission: {resubmission_count}")
        logger.info(f"Claims not eligible for resubmission: {ineligible_count}")
        logger.info(f"Rejected records: {len(self.rejected_records)}")
        
        # Save results
        self.save_results()
    
    def save_results(self) -> None:
        """Save resubmission candidates to JSON file"""
        try:
            with open('resubmission_candidates.json', 'w') as f:
                json.dump(self.resubmission_candidates, f, indent=2)
            logger.info("Resubmission candidates saved to resubmission_candidates.json")
            
            # Also save rejected records for debugging
            with open('rejected_records.json', 'w') as f:
                json.dump(self.rejected_records, f, indent=2)
            logger.info("Rejected records saved to rejected_records.json")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


# Main execution
if __name__ == "__main__":
    # Create sample data files
    alpha_data = """claim_id,patient_id,procedure_code,denial_reason,submitted_at,status
A123,P001,99213,Missing modifier,2025-07-01,denied
A124,P002,99214,Incorrect NPI,2025-07-10,denied
A125,,99215,Authorization expired,2025-07-05,denied
A126,P003,99381,None,2025-07-15,approved
A127,P004,99401,Prior auth required,2025-07-20,denied"""
    
    beta_data = [
        {
            "id": "B987",
            "member": "P010",
            "code": "99213",
            "error_msg": "Incorrect provider type",
            "date": "2025-07-03T00:00:00",
            "status": "denied"
        },
        {
            "id": "B988",
            "member": "P011",
            "code": "99214",
            "error_msg": "Missing modifier",
            "date": "2025-07-09T00:00:00",
            "status": "denied"
        },
        {
            "id": "B989",
            "member": "P012",
            "code": "99215",
            "error_msg": None,
            "date": "2025-07-10T00:00:00",
            "status": "approved"
        },
        {
            "id": "B990",
            "member": None,
            "code": "99401",
            "error_msg": "incorrect procedure",
            "date": "2025-07-01T00:00:00",
            "status": "denied"
        }
    ]
    
    # Write sample data to files
    with open('emr_alpha.csv', 'w') as f:
        f.write(alpha_data)
    
    with open('emr_beta.json', 'w') as f:
        json.dump(beta_data, f, indent=2)
    
    # Process the claims
    processor = ClaimProcessor()
    processor.process_claims('emr_alpha.csv', 'emr_beta.json')
