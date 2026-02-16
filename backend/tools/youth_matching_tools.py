"""Custom tools for the youth opportunity matching platform."""

import json
from agno.tools import Toolkit
from agno.utils.log import logger


class YouthMatchingTools(Toolkit):
    """Toolkit for youth matching platform operations."""

    def __init__(self, db_connection=None, **kwargs):
        self.db = db_connection
        tools = [
            self.post_job_requirement,
            self.send_match_notification,
            self.update_youth_availability,
            self.get_match_statistics,
        ]
        super().__init__(name="youth_matching_tools", tools=tools, **kwargs)

    def post_job_requirement(
        self,
        company: str,
        position: str,
        required_skills: str,
        location: str,
        qualifications: str,
        job_type: str,
    ) -> str:
        """Post a new job requirement from a company.

        Args:
            company: Company name
            position: Job title/position
            required_skills: Comma-separated required skills
            location: Job location
            qualifications: Required qualifications
            job_type: full-time, part-time, or internship

        Returns:
            Confirmation with job posting details
        """
        logger.info(f"New job posted: {position} at {company}")
        job = {
            "company": company,
            "position": position,
            "skills": required_skills,
            "location": location,
            "qualifications": qualifications,
            "type": job_type,
            "status": "active",
        }
        return json.dumps({"status": "posted", "job": job})

    def send_match_notification(
        self,
        youth_name: str,
        job_position: str,
        company: str,
        match_reason: str,
    ) -> str:
        """Send a notification about a potential match.

        Args:
            youth_name: Name of the matched youth
            job_position: The job position matched
            company: The company offering the position
            match_reason: Why this match was made

        Returns:
            Notification status
        """
        logger.info(f"Match notification: {youth_name} -> {job_position} at {company}")
        return json.dumps(
            {
                "status": "notified",
                "match": {
                    "youth": youth_name,
                    "position": job_position,
                    "company": company,
                    "reason": match_reason,
                },
            }
        )

    def update_youth_availability(
        self,
        youth_name: str,
        available: bool,
        notes: str = "",
    ) -> str:
        """Update a youth's job availability status.

        Args:
            youth_name: Name of the youth
            available: Whether they are currently available
            notes: Optional notes about availability

        Returns:
            Updated availability status
        """
        return json.dumps(
            {"youth": youth_name, "available": available, "notes": notes}
        )

    def get_match_statistics(self) -> str:
        """Get current matching statistics and metrics.

        Returns:
            Statistics about matches, placements, and pending applications
        """
        return json.dumps(
            {
                "total_youth": 32,
                "total_employers": 37,
                "active_matches": 12,
                "placed": 8,
                "pending": 4,
            }
        )
