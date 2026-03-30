from cce.identity.resolver import resolve_project_id


def test_resolve_git_repo_is_stable():
    # This test runs inside the project repo, so git root is detectable
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    id1 = resolve_project_id(project_root)
    id2 = resolve_project_id(project_root)
    assert id1 == id2
    assert len(id1) == 16


def test_resolve_same_repo_from_subdir():
    import os
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    subdir = os.path.join(root, "cce")
    id_root = resolve_project_id(root)
    id_sub = resolve_project_id(subdir)
    assert id_root == id_sub


def test_resolve_named_workspace_is_stable():
    id1 = resolve_project_id("my-project-session")
    id2 = resolve_project_id("my-project-session")
    assert id1 == id2
    assert len(id1) == 16


def test_resolve_different_names_differ():
    assert resolve_project_id("project-a") != resolve_project_id("project-b")
