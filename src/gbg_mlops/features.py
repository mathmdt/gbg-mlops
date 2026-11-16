"""Preprocessing shared by training and the API.

Single source of truth for channel order, units, windowing and normalisation,
so that a ride is transformed identically everywhere. Parameters are defined in
`CONTRACTS.md` (step 2); the batch implementation is completed by the pipelines
owner.
"""
