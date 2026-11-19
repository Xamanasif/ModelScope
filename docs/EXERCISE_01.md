# Your first contribution: understand and challenge the split
Before Claude completes the application:
1. Read split_customers and explain why it stratifies on the target.
2. Explain why the code allocates integer partition sizes up front rather than
   repeatedly using fractional test sizes. Try 400 and 403 customers.
3. Run test_split_is_disjoint_complete_and_reproducible.
4. Change the split seed and show that customer membership changes.
5. Explain why unique IDs are checked before splitting.
6. Explain why this split cannot measure future-cohort performance.
7. Add a test that no customer appears twice within a partition.
Write your explanation and what you changed in docs/MY_CONTRIBUTION.md.
