+++
title =  "Hosting a Hugo-Generated Blog on Google Cloud Storage"
tags = ["hugo", "gcp"]
date = "2024-07-18"
+++

# Test
Some content
{{< figure src="/images/gcp-blog/access-control.png" title="Setting access control settings" >}}

Some notes:
* Provisioning the cert for your load balancer takes time. Do it first!
  * Took me about 40 minutes, but could be longer
* APIs used in this solution (maybe enable them upfront?)
  * compute
  * storage
  * build