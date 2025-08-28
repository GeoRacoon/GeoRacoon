










def check_rank_deficiency(array, return_by_issue_type: bool=False) -> dict[int, str] | dict[str, list[int]]:
    # TODO: is_needed - no_work - is_tested - usedin_linfit
    """Check if matrix is rank deficient and extract the dependent columns (linear combination of other columns.
    Returns a dictionary with column (key) and issue description (value). Lenght of dictionary is rank-deficiency + 1,
    Empyt dictionary indicates that no rank deficiency was detected

    Parameters
    ----------
    array : np.ndarray
        Matrix to check for rank deficiency
    return_by_issue_type: bool
        If desired, a nested dictionary may be returned separating the type of issue:
        "all_zero" and "linear dependent"
    """
    # is_needed
    # needs_work (formatting)
    # is_tested
    # usedin_linfit
    all_zero_cols = {}
    rank_deficient_cols = {}
    _, num_columns = array.shape
    rank = np.linalg.matrix_rank(array)

    if rank == num_columns:
        return dict()

    for col in range(num_columns):
        column_vector = array[:, col]

        if np.all(column_vector == 0):
            all_zero_cols[col] = "All zero column"
        else:
            # drop focus column
            sub_array = np.delete(array, col, axis=1)

            # does removing a column increase the rank?
            if np.linalg.matrix_rank(sub_array) == rank:
                rank_deficient_cols[col] = "Linear dependent column"

    if return_by_issue_type:
        return dict(linear_dependent=[l for l in rank_deficient_cols.keys()],
                    all_zero=[z for z in all_zero_cols.keys()])
    else:
        return {**rank_deficient_cols, **all_zero_cols}
