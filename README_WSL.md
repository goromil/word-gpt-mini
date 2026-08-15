Forwarding WSL2 ports back to Windows


VS Code's Remote - WSL doesn't provide SSH-style tunneling for arbitrary ports. 

It uses a proxy mechanism, but I'm not sure it can forward from WSL2 back to Windows loopback. The most practical solutions are either reverting to NAT mode with proper firewall configuration, or running llama-server directly inside WSL2.


Your `.vscode/settings.json` is minimal. Let me outline the **practical options**:

## Option A: Revert to NAT mode (simplest fix)

```powershell
# Remove mirrored mode
# Edit C:\Users\gorom\.wslconfig:
@"
[wsl2]
networkingMode=nat
"@ | Out-File -Encoding ASCII $env:USERPROFILE\.wslconfig

# Allow WSL2 subnet through firewall
New-NetFirewallRule -DisplayName "WSL2 llama-server 8080" -Direction Inbound -Protocol TCP -LocalPort 8080 -RemoteAddress 172.16.0.0/12 -Action Allow

# Restart WSL
wsl --shutdown
```

Then in WSL2, use:
```bash
# The vEthernet adapter IP from your ipconfig above
curl http://172.24.16.1:8080/v1/chat/completions ...
```

## Option B: Run llama-server inside WSL2

Move the model and `llama-server` binary into WSL2. Then everything is local and `localhost:8080` works natively.

## Option C: SSH tunnel (requires OpenSSH on Windows)

```powershell
# Enable OpenSSH server on Windows first
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd
```

Then from WSL2:
```bash
ssh -L 8080:127.0.0.1:8080 localhost
```

---

**Option A is the quickest fix** — it's just a config change and a firewall rule. Which approach do you want to go with?

---
