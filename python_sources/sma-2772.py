
    import pandas as pd 
cases = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
import plotly.offline as py
import plotly.express as px


py.init_notebook_mode(connected=True)

grp = cases.groupby(['ObservationDate', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered'].max()
grp = grp.reset_index()
grp['Date'] = pd.to_datetime(grp['ObservationDate'])
grp['Date'] = grp['Date'].dt.strftime('%m/%d/%Y')
grp['Active'] = grp['Confirmed'] - grp['Recovered'] - grp['Deaths']
grp['Country'] =  grp['Country/Region']

fig = px.choropleth(grp, locations="Country", locationmode='country names', 
                     color="Confirmed", hover_name="Country/Region",hover_data = [grp.Recovered,grp.Deaths,grp.Active],projection="mercator",
                     animation_frame="Date",width=1000, height=700,
                     color_continuous_scale='Reds',
                     range_color=[1000,50000],

                     title='World Map of Coronavirus')

fig.update(layout_coloraxis_showscale=True)
py.offline.iplot(fig)


{
   "schemaVersion": 2,
   "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
   "config": {
      "mediaType": "application/vnd.docker.container.image.v1+json",
      "size": 28903,
      "digest": "sha256:9714ae1a7ae02410d11d6f49afb1dbf402b8cee4cbc6a6b60244543c25343f5b"
   },
   "layers": [
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 45339314,
         "digest": "sha256:c5e155d5a1d130a7f8a3e24cee0d9e1349bff13f90ec6a941478e558fde53c14"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 95104141,
         "digest": "sha256:86534c0d13b7196a49d52a65548f524b744d48ccaf89454659637bee4811d312"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1571501372,
         "digest": "sha256:5764e90b1fae3f6050c1b56958da5e94c0d0c2a5211955f579958fcbe6a679fd"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1083072,
         "digest": "sha256:ba67f7304613606a1d577e2fc5b1e6bb14b764bcc8d07021779173bcc6a8d4b6"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 528,
         "digest": "sha256:d81ed61a4422a354e1b0afef7da163290db14ce2294efd5b70a04fffde7cadc7"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 460,
         "digest": "sha256:48da9ce5949eb799b8df23bc10b57ab46dcf95d797a9c4025389b9cc6c065065"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 13118033,
         "digest": "sha256:c2602205c5a3e9c193ff2658c94293d1b91b4b1d87ec65ed0b2590ee4d887659"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 568822119,
         "digest": "sha256:5c469001e55edd105b337997a62680b24cb9d3b0c9876fc3d0022c7ea685cf7e"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 560507262,
         "digest": "sha256:360c2e45ee312c29ee3168505b1248d95c7f4fc44489710e4052d0a56696cca8"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 111202686,
         "digest": "sha256:1660597b7dafa84a8968cbf50a1dc6ecd7c3dc0ef9191ef1fde8430704fd58fa"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 95925401,
         "digest": "sha256:6712b77ef8f6f2f6bef65937df0ba8f5db336f3ec6147add871adf1f2a777fc3"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 142490147,
         "digest": "sha256:1f866f3e89a8ef45ed0ca14115281646c182e2b8bd57370dffbe1693299a10a8"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1057757366,
         "digest": "sha256:0121a2ebc0c03d84d10d155a9d638ae25cf5476a4e5140471de922d5b075f846"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 75007521,
         "digest": "sha256:5a78be6b6647746fe108c25ab6e9b949a0ea2964a9c180108a7cdbebbca89679"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 75685490,
         "digest": "sha256:51cbd07ed997dfc30dcc397cd3e52468810d89e945328ca43f41b7c5dd7c303b"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 89502363,
         "digest": "sha256:c8456022b1d65c496be61f9bc1bde90caa8d5603d50a2cd4bfb8b29801017bd5"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 436174950,
         "digest": "sha256:d5f7d29b47bd215698b1199a5643bd305214ecac3970b655f6aeefbf343da102"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 44168775,
         "digest": "sha256:f15c6804a4a7e66375b7cb9f50019df8662b45f3cb5b59792f42d7865f064d97"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 107815431,
         "digest": "sha256:56575b25c4cbbb29937cf91068c0bb421b981f0c0678ec5b34ef38f3888d043a"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 957451081,
         "digest": "sha256:818db54c9479612f1c0a87f28bf8b8821c07f1cffbad5b00f9fda7697d3aef29"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 587181409,
         "digest": "sha256:a8175488bbb16afb1764dcc3e65549d6a0f39048a80faa095a48d0a03f7b27e9"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 21702994,
         "digest": "sha256:8ee73aa94042bb09fd3e0a00c0f8411b20c0733fa386390de4b7e8df6495be40"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 215780690,
         "digest": "sha256:72a9f2297a16a0bbe8b53f2de8e039abdbf94cb93cf375d9878cfb7dd59381ad"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 19929,
         "digest": "sha256:9c547216aa9d06286f680e1af72752311ad0fe3ea0b92c42d20797bc496808ce"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 85350347,
         "digest": "sha256:895298d3438105d79a924eb54e24589c2ce8a430d242894ddf3be3c55a0c2d1d"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 3267,
         "digest": "sha256:a6e7a8217d2bbb8c4a4bd5ef28f51751927c2ef27788d0f9dbac0b9a9288ac68"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 2167,
         "digest": "sha256:511e86077eb1b34bbcb2ef4bda662ede644e4cf051670862825c011aa1312fc3"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 1270,
         "digest": "sha256:546afd8856dc4f449f8148e027301ddb8156a23f1e2e1f6cb429d18a8d8086fd"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 644,
         "digest": "sha256:ef2d043f73afd238054b54d8d05451cf92a231254c5fb6097668ca2cd342965d"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 2051,
         "digest": "sha256:c0487916fd2193c0883b86ba1a406b06cb5810c958b3da7179570b2711a49226"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 876,
         "digest": "sha256:b6bff9871f743216b17f1db78f284b88f000c2d026edbf559b60beae82f9d27b"
      },
      {
         "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
         "size": 212,
         "digest": "sha256:00825515b888e084e28dfb5659b46acb3584b23c6cdb4e1fb9ec8ff44714d363"
      }
   ]
}

# Any results you write to the current directory are saved as output.