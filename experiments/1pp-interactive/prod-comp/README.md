# Interactive experiment

NOTE: For the avatars to show up in the chat, comment out 

```js
if (!avatar.startsWith("http")) {
avatarImage = /* @__PURE__ */ React6.createElement("div", { className: "inline-block h-9 w-9 rounded-full" }, avatar);
}
```

in `chunk-J6LPACOK.js` in `node_modules/@empirica/core/dist`. This is because `player.get("avatar")` isn't the full image tag, just the path (and this is so that we can also use `player.get("avatar")` in `Avatar.jsx`)