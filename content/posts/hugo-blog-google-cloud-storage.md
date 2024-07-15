+++
title =  "Hosting a Hugo-Generated Blog on Google Cloud Storage"
tags = ["hugo", "gcp"]
date = "2024-07-18"
+++

# Test
Some content
![Setting access control settings](/images/gcp-blog/access-control.png)

Some notes:
* Provisioning the cert for your load balancer takes time. Do it first!
* APIs used in this solution (maybe enable them upfront?)
  * compute
  * storage
  * build