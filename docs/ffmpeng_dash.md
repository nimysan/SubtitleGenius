# 一个例子

https://reference.dashif.org/dash.js/v4.4.0/samples/captioning/caption_vtt.html

## mpd是可以用这样的内容

```xml
<AdaptationSet mimeType="text/vtt" lang="en"> 
    <Representation id="caption_en" bandwidth="256">
        <BaseURL>https://dash.akamaized.net/akamai/test/caption_test/ElephantsDream/ElephantsDream_en.vtt</BaseURL> 
    </Representation>
<AdaptationSet/>
```
