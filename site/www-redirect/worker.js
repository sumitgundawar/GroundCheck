// Permanently redirects www.groundcheckhealth.com to groundcheckhealth.com,
// keeping the path and query string.
const CANONICAL_HOST = "groundcheckhealth.com";

export default {
  fetch(request) {
    const url = new URL(request.url);
    url.hostname = CANONICAL_HOST;
    url.protocol = "https:";
    url.port = "";
    return Response.redirect(url.toString(), 301);
  },
};
