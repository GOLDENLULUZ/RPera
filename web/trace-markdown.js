import createDOMPurify from "dompurify";
import MarkdownIt from "markdown-it";

const markdown = new MarkdownIt({ html: false, linkify: false, breaks: true });
markdown.disable("image");
const purifier = createDOMPurify(window);

export function render(text) {
    return purifier.sanitize(markdown.render(text), {
        USE_PROFILES: { html: true },
        FORBID_TAGS: ["img"],
        FORBID_ATTR: ["style"],
    });
}
