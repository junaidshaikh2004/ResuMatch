def get_or_create_session_key(request):
    """
    Every visitor needs a stable session key even before Django would
    normally create one (Django only creates a session key once something
    is written to the session). We force that here so a brand-new visitor
    can immediately be tied to a Resume row.
    """
    if not request.session.session_key:
        request.session.create()
    return request.session.session_key
