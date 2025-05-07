#!/usr/bin/env python

# Simple test script to verify overqualified candidates are correctly ranked

profiles = [
    {
        "profile_id": "1001",
        "is_overqualified": True,
        "overall_match": {"percentage": 95}
    },
    {
        "profile_id": "1002",
        "is_overqualified": False,
        "overall_match": {"percentage": 85}
    },
    {
        "profile_id": "1003",
        "is_overqualified": False,
        "overall_match": {"percentage": 75}
    },
    {
        "profile_id": "1004",
        "is_overqualified": True,
        "overall_match": {"percentage": 90}
    },
    {
        "profile_id": "1005",
        "is_overqualified": False,
        "overall_match": {"percentage": 95}
    }
]

print("Original profiles:")
for i, profile in enumerate(profiles):
    print(f"{i+1}. ID: {profile['profile_id']}, Overqualified: {profile['is_overqualified']}, Match: {profile['overall_match']['percentage']}")

# Sort by old method
old_sorted = sorted(
    profiles,
    key=lambda x: x.get("overall_match", {}).get("percentage", 0),
    reverse=True
)

print("\nSorted by old method (only by percentage):")
for i, profile in enumerate(old_sorted):
    print(f"{i+1}. ID: {profile['profile_id']}, Overqualified: {profile['is_overqualified']}, Match: {profile['overall_match']['percentage']}")

# Sort by new method
new_sorted = sorted(
    profiles,
    key=lambda x: (
        # Sort overqualified candidates after non-overqualified candidates
        x.get("is_overqualified", False),
        # Then by overall match percentage (descending)
        -x.get("overall_match", {}).get("percentage", 0)
    )
)

print("\nSorted by new method (overqualified last, then by percentage):")
for i, profile in enumerate(new_sorted):
    print(f"{i+1}. ID: {profile['profile_id']}, Overqualified: {profile['is_overqualified']}, Match: {profile['overall_match']['percentage']}")

# Assign ranks
for i, profile in enumerate(new_sorted):
    profile["rank"] = i + 1

print("\nFinal rankings:")
for profile in sorted(new_sorted, key=lambda x: x["profile_id"]):
    print(f"ID: {profile['profile_id']}, Rank: {profile['rank']}, Overqualified: {profile['is_overqualified']}, Match: {profile['overall_match']['percentage']}") 