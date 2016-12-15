class WebPage(object):

    """Define WebPage object based on web-page-link"""

    def __init__(self, title, link, content, label=None):
        self.title = title
        self.link = link
        self.label = label
        self.content = content
