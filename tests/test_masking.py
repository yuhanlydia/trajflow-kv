def test_assistant_span_starts_after_prompt():
    # Regression guard for chat templates that append EOS after the action.
    prompt_ids = [10, 11, 12]
    full_ids = [10, 11, 12, 42, 43, 99]
    labels = full_ids[:]
    labels[:len(prompt_ids)] = [-100] * len(prompt_ids)
    assert labels == [-100, -100, -100, 42, 43, 99]
