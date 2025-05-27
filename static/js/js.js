$(document).ready(function () {

    function loadTable(callback) {
        var container = $(".data_phong_tb");
        container.empty();
        container.append('<div>Đang tải dữ liệu...</div>');

        $.post('#', { action: 'get_infor' }, function (data) {
            container.empty();
            if (data.length > 0) {
                data.forEach(function (phong) {
                    var phongDiv = $("<div class='phong'></div>");
                    phongDiv.append($("<div></div>").html(
                        `<div class="tt_phong">
                            <h3 style="text-align: center;"> Thông Tin Phòng Học </h3>
                            <h4>Mã Phòng Học: ${phong.maphong}</h4>
                            <h4>Tên Phòng Học: ${phong.tenphong}</h4>
                            <h4>Trạng Thái Phòng Học : ${statusPhong(phong.ttphong)}</h4>
                        </div>
                        <div class="change">
                            <button class="button_phong" id="add-btn" onclick="them_phong()">Thêm phòng</button>
                            <button class="button_phong" id="edit-btn" onclick="sua_phong('${phong.maphong}', this)">Sửa</button>
                            <button class="button_phong" id="delete-btn" onclick="xoa_phong('${phong.maphong}', this)">Xóa</button>
                        </div>
                    `));

                    var thietBiListDiv = $("<div class='thiet_bi_list'></div>");
                    thietBiListDiv.append('<strong style="display: flex; justify-content: space-around;">Danh Sách Thiết Bị:</strong>');
                    thietBiListDiv.append($("<div class='button-add'></div>").html(`<button class="button_tb" id="add-btn" onclick="them_tb('${phong.maphong}', this)">Thêm thiết bị</button>`));

                    if (phong.thiet_bi_list && phong.thiet_bi_list.length > 0) {
                        phong.thiet_bi_list.forEach(function (thietBi) {
                            var thietBiDiv = $("<div class='thietbi'></div>");
                            thietBiDiv.append($('<div class="tt_thietbi"></div>').text("Mã Thiết Bị: " + thietBi.matb + ", Tên: " + thietBi.tentb + ", Trạng Thái: " + status(thietBi.tttbi)));
                            thietBiDiv.append($('<div class="tb_nut"></div>').html(
                                `<button class="button_tb" id="edit-btn" onclick="sua_tb('${thietBi.matb}', this)">Sửa</button>
                                     <button class="button_tb" id="delete-btn" onclick="xoa_tb('${thietBi.matb}', this)">Xóa</button>`
                            ));
                            thietBiListDiv.append(thietBiDiv);
                        });
                    } else {
                        thietBiListDiv.append("<div>Không có thiết bị nào trong phòng này.</div>");
                    }

                    phongDiv.append(thietBiListDiv);
                    container.append(phongDiv);
                });
            } else {
                container.append('<div>Không có dữ liệu nào.</div>');
            }
            if (callback) callback();
        }, 'json');
    }
   
    function status(tttb) {
        tttb = parseInt(tttb);
        switch (tttb) {
            case 1: return "Thiết bị đang dùng";
            case 2: return "Thiết bị không dùng";
            case 3: return "Thiết bị hỏng";
            default: return "Trạng thái không rõ";
        }
    }

    function statusPhong(ttphong) {
        ttphong = parseInt(ttphong);
        switch (ttphong) {
            case 1: return "Phòng đang học";
            case 2: return "Phòng không học";
            case 3: return "Phòng đang sửa chữa";
            default: return "Trạng thái không rõ";
        }
    }


    window.them_phong = function () {
        var html =
            `<form id="addDeviceForm">
                         <div class="mb-3">
                            <label for="maphongInput" class="form-label">Mã phòng</label>
                            <input type="text" class="form-control" id="maphongInput" placeholder="Nhập mã phòng">
                        </div>
                         <div class="mb-3">
                            <label for="tenphongInput" class="form-label">Tên phòng</label>
                            <input type="text" class="form-control" id="tenphongInput" placeholder="Nhập tên phòng">
                        </div>
                         <div class="mb-3">
                            <label for="ttphongInput" class="form-label">Trạng thái phòng</label>
                            <select class="form-control" id="ttphongInput">
                                <option value="1">Phòng đang học</option>
                                <option value="2">Phòng trống</option>
                                <option value="3">Phòng đang sửa chữa</option>
                            </select>
                        </div>
                    </form>`;

        var confirmBox = $.confirm({
            title: 'Thêm Phòng',
            content: html,
            boxWidth: '30%',
            theme: 'material',
            buttons: {
                add: {
                    text: 'Thêm',
                    btnClass: 'btn-primary',
                    action: function () {
                        var maphong = $('#maphongInput').val();
                        var tenphong = $('#tenphongInput').val();
                        var ttphong = $('#ttphongInput').val();

                        if (!maphong || !tenphong || !ttphong) {
                            $.alert('Vui lòng nhập đầy đủ thông tin!');
                            return false;
                        }

                        $.post('#', {
                            action: 'add_phong',
                            maphong: maphong,
                            tenphong: tenphong,
                            ttphong: ttphong
                        }, function (data) {
                            if (data.ok) {
                                $.alert('Đã Thêm Thành Công');
                                them_phong_dom(maphong, tenphong, ttphong);
                                confirmBox.close();
                            } else {
                                $.alert('Thêm Thất Bại');
                            }
                        }, 'json');
                        return false;
                    }
                },
                cancel: {
                    text: 'Hủy',
                    btnClass: 'btn-secondary',
                }
            }
        });
    };

    function them_phong_dom(maphong, tenphong, ttphong) {
        var phongDiv = $("<div class='phong'></div>");
        phongDiv.append($("<div></div>").html(
            `<div class="tt_phong">
                            <h3 style="text-align: center;"> Thông Tin Phòng Học </h3>
                            <h4>Mã Phòng Học: ${maphong}</h4>
                            <h4>Tên Phòng Học: ${tenphong}</h4>
                            <h4>Trạng Thái Phòng Học : ${statusPhong(ttphong)}</h4>
                        </div>
                        <div class="change">
                            <button class="button_phong" id="add-btn" onclick="them_phong()">Thêm phòng</button>
                            <button class="button_phong" id="edit-btn" onclick="sua_phong('${maphong}', this)">Sửa</button>
                            <button class="button_phong" id="delete-btn" onclick="xoa_phong('${maphong}', this)">Xóa</button>
                        </div>
                    `));

        var thietBiListDiv = $("<div class='thiet_bi_list'></div>");
        thietBiListDiv.append('<strong style="display: flex; justify-content: space-around;">Danh Sách Thiết Bị:</strong>');
        thietBiListDiv.append($("<div class='button-add'></div>").html(`<button class="button_tb" id="add-btn" onclick="them_tb('${maphong}', this)">Thêm thiết bị</button>`));
        phongDiv.append(thietBiListDiv);
        $(".data_phong_tb").append(phongDiv);
    }


    window.sua_phong = function (maphong, button) {
        var phongDiv = $(button).closest('.phong');
        var currentTenPhong = phongDiv.find('.tt_phong h4').eq(1).text().replace("Tên Phòng Học: ", "").trim();
        var currentTtPhong = phongDiv.find('.tt_phong h4').eq(2).text().replace("Trạng Thái Phòng Học: ", "").trim();

        var currentStatusValue;
        if (currentTtPhong.includes("đang học")) {
            currentStatusValue = "1";
        } else if (currentTtPhong.includes("không học")) {
            currentStatusValue = "2";
        } else if (currentTtPhong.includes("đang sửa chữa")) {
            currentStatusValue = "3";
        }

        var html = `
        <form id="editPhongForm">
            <div class="mb-3">
                <label for="tenphongEditInput" class="form-label">Tên Phòng</label>
                <input type="text" class="form-control" id="tenphongEditInput" value="${currentTenPhong}">
            </div>
            <div class="mb-3">
                <label for="ttphongEditInput" class="form-label">Trạng Thái Phòng</label>
                <select class="form-select" id="ttphongEditInput">
                    <option value="1" ${currentStatusValue === "1" ? 'selected' : ''}>Phòng đang học</option>
                    <option value="2" ${currentStatusValue === "2" ? 'selected' : ''}>Phòng không học</option>
                    <option value="3" ${currentStatusValue === "3" ? 'selected' : ''}>Phòng đang sửa chữa</option>
                </select>
            </div>
        </form>
    `;

        $.confirm({
            title: 'Chỉnh Sửa Phòng',
            content: html,
            boxWidth: '30%',
            buttons: {
                save: {
                    text: 'Sửa',
                    btnClass: 'btn-primary',
                    action: function () {
                        var newTenphong = $('#tenphongEditInput').val();
                        var newTtphong = $('#ttphongEditInput').val();

                        $.post('#', { action: 'update_phong', maphong: maphong, tenphong: newTenphong, ttphong: newTtphong }, function (data) {
                            if (data.ok) {
                                $.alert('Chỉnh Sửa Thành Công');
                                phongDiv.find('.tt_phong h4').eq(1).text("Tên Phòng Học: " + newTenphong);
                                phongDiv.find('.tt_phong h4').eq(2).text("Trạng Thái Phòng: " + statusPhong(newTtphong));
                            } else {
                                $.alert('Chỉnh Sửa Thất Bại');
                            }
                        }, 'json');
                    }
                },
                cancel: {
                    text: 'Hủy',
                    btnClass: 'btn-secondary',
                }
            }
        });
    };


    window.xoa_phong = function (maphong, button) {
        $.confirm({
            title: 'Xóa Phòng',
            content: 'Bạn có chắc chắn muốn xóa phòng này?',
            buttons: {
                confirm: {
                    text: 'Xóa',
                    btnClass: 'btn-danger',
                    action: function () {
                        $.post('#', { action: 'delete_phong', maphong: maphong }, function (data) {
                            if (data.ok) {
                                $.alert('Đã Xóa Thành Công');
                                $(button).closest('.phong').remove();
                            } else {
                                $.alert('Xóa thất bại');
                            }
                        }, 'json');
                    }
                },
                cancel: {
                    text: 'Hủy',
                    btnClass: 'btn-secondary',
                }
            }
        });
    };
    window.them_tb = function (maphong, button) {
        var html =
            `<form id="addDeviceForm">
                             <div class="mb-3">
                                <label class="form-label">Mã Phòng: ${maphong}</label>
                                <label for="matbInput" class="form-label">Mã Thiết Bị</label>
                                <input type="text" class="form-control" id="matbInput" placeholder="Nhập mã thiết bị">
                            </div>
                             <div class="mb-3">
                                <label for="tentbInput" class="form-label">Tên Thiết Bị</label>
                                <input type="text" class="form-control" id="tentbInput" placeholder="Nhập tên thiết bị">
                            </div>
                             <div class="mb-3">
                                <label for="tttbiInput" class="form-label">Trạng Thái Thiết Bị</label>
                                <select class="form-control" id="tttbiInput">
                                    <option value="1">Thiết bị đang dùng</option>
                                    <option value="2">Thiết bị không dùng</option>
                                    <option value="3">Thiết bị hỏng</option>
                                </select>
                            </div>
                        </form>`;

        var confirmBox = $.confirm({
            title: 'Thêm Thiết Bị',
            content: html,
            boxWidth: '30%',
            theme: 'material',
            buttons: {
                add: {
                    text: 'Thêm',
                    btnClass: 'btn-primary',
                    action: function () {
                        var matb = $('#matbInput').val();
                        var tentb = $('#tentbInput').val();
                        var tttbi = $('#tttbiInput').val();

                        if (!matb || !tentb || !tttbi) {
                            $.alert('Vui lòng nhập đầy đủ thông tin!');
                            return false;
                        }

                        $.post('#', {
                            action: 'add_tb',
                            maphong: maphong,
                            matb: matb,
                            tentb: tentb,
                            tttbi: tttbi
                        }, function (data) {
                            if (data.ok) {
                                $.alert('Đã Thêm Thành Công');
                                them_thiet_bi_dom(maphong, matb, tentb, tttbi, button);
                                confirmBox.close();
                            } else {
                                $.alert('Thêm Thất Bại');
                            }
                        }, 'json');
                        return false;
                    }
                },
                cancel: {
                    text: 'Hủy',
                    btnClass: 'btn-secondary',
                }
            }
        });
    };

    function them_thiet_bi_dom(maphong, matb, tentb, tttbi, button) {
        var thietBiDiv = $("<div class='thietbi'></div>");
        thietBiDiv.append($('<div class="tt_thietbi"></div>').text("Mã Thiết Bị: " + matb + ", Tên: " + tentb + ", Trạng Thái: " + status(tttbi)));
        thietBiDiv.append($('<div class="tb_nut"></div>').html(
            `<button class="button_tb" id="edit-btn" onclick="sua_tb('${matb}', this)">Sửa</button>
                <button class="button_tb" id="delete-btn" onclick="xoa_tb('${matb}', this)">Xóa</button>`
        ));
        $(button).closest('.thiet_bi_list').append(thietBiDiv);
    }


    window.sua_tb = function (matb, button) {
        var container = $(button).closest('.thietbi');
        var currentInfo = container.find('div').eq(0).text();
        var currentName = currentInfo.split(", Tên: ")[1].split(", Trạng Thái: ")[0];
        var currentStatus = currentInfo.split(", Trạng Thái: ")[1];

        var html = `
                <form id="editDeviceForm">
                    <div class="mb-3">
                        <label for="tentbEditInput" class="form-label">Tên Thiết Bị</label>
                        <input type="text" class="form-control" id="tentbEditInput" value="${currentName}">
                    </div>
                    <div class="mb-3">
                        <label for="tttbiEditInput" class="form-label">Trạng Thái Thiết Bị</label>
                        <select class="form-select" id="tttbiEditInput">
                            <option value="1" ${currentStatus === "Thiết bị đang dùng" ? 'selected' : ''}>Thiết bị đang dùng</option>
                            <option value="2" ${currentStatus === "Thiết bị không dùng" ? 'selected' : ''}>Thiết bị không dùng</option>
                            <option value="3" ${currentStatus === "Thiết bị hỏng" ? 'selected' : ''}>Thiết bị hỏng</option>
                        </select>
                    </div>
                </form>
            `;

        $.confirm({
            title: 'Chỉnh Sửa Thiết Bị',
            content: html,
            boxWidth: '30%',
            buttons: {
                save: {
                    text: 'Sửa',
                    btnClass: 'btn-primary',
                    action: function () {
                        var newTentb = $('#tentbEditInput').val();
                        var newTttbi = $('#tttbiEditInput').val();

                        $.post('#', { action: 'update_tb', matb: matb, tentb: newTentb, tttbi: newTttbi }, function (data) {
                            if (data.ok) {
                                $.alert('Chỉnh Sửa Thành Công');
                                container.find('div').eq(0).text("Mã Thiết Bị: " + matb + ", Tên: " + newTentb + ", Trạng Thái: " + status(newTttbi));
                            } else {
                                $.alert('Chỉnh Sửa Thất Bại');
                            }
                        }, 'json');
                    }
                },
                cancel: {
                    text: 'Hủy',
                    btnClass: 'btn-secondary',
                }
            }
        });
    };

    window.xoa_tb = function (matb, button) {
        $.confirm({
            title: 'Xóa Thiết Bị',
            content: 'Bạn có chắc chắn muốn xóa thiết bị này?',
            boxWidth: '30%',
            buttons: {
                confirm: {
                    text: 'Xóa',
                    btnClass: 'btn-danger',
                    action: function () {
                        $.post('#', { action: 'delete_tb', matb: matb }, function (data) {
                            if (data.ok) {
                                $.alert('Đã Xóa Thành Công');
                                $(button).closest('.thietbi').remove();
                            } else {
                                $.alert('Xóa thất bại');
                            }
                        }, 'json');
                    }
                },
                cancel: {
                    text: 'Hủy',
                    btnClass: 'btn-secondary',
                }
            }
        });
    }

    function dangky() {
        const name = $("#nameInput").val().trim();
        const dienthoai = $("#dienthoaiInput").val().trim();
        const maso = $("#masoInput").val().trim();
        const gmail = $("#gmailInput").val().trim();
        const diachi = $("#diachiInput").val().trim();
        const username = $("#usernameInput").val().trim();
        var PassDK = $("#PassInput").val().trim();
        const RepeatPass = $("#RepeatPassInput").val().trim();

        if (!name || !dienthoai || !maso || !gmail || !diachi || !username || !PassDK || !RepeatPass) {
            alert("Vui lòng điền đầy đủ thông tin.");
            return;
        }

        if (PassDK !== RepeatPass) {
            alert("Mật khẩu không khớp, vui lòng kiểm tra lại.");
            return;
        }
        var Pass = CryptoJS.SHA256(PassDK).toString();
        console.log(typeof(Pass));
        console.log(Pass);
        $.post('#', {
            action: 'dangky',
            user: username,
            pass: Pass,
            name: name,
            maso: maso,
            gmail: gmail,
            dienthoai: dienthoai,
            diachi: diachi }, function (data) {
            if (data.ok == 1) {
                salt_value = data.salt;
                if (salt_value != '') {
                    $(location).attr('href', 'dangnhap.html');
                } else {
                    alert("Tài khoản chưa được đăng ký!");
                }
            } else {
                alert("Đăng ký thất bại");
            }
        }, 'json');
    }


    var loginAttempts = 0;

    function get_salt(callback) {
        var username = $("#usernameInputDN").val();

        $.post('#', { action: 'get_salt', uid: username }, function (data) {
            if (data.ok == 1) {
                salt_value = data.salt;
                if (salt_value != '') {
                    callback(salt_value);
                } else {
                    alert("Tài khoản chưa được đăng ký!");
                }
            } else {
                loginAttempts++;
                if (loginAttempts >= 3) {
                    $("#captchaSection").show();
                    $("#captchaImage").attr("src", "api.aspx?action=generate_captcha");
                }
                alert("Tài khoản không tồn tại");
                loginAttempts++;
            }
        }, 'json');
    }
    function performLogin(salt_value) {

        var username = $("#usernameInputDN").val();
        var password = $("#PassInputDN").val();

        if (!username || !password) {
            alert("Vui lòng điền tên đăng nhập và mật khẩu.");
            return;
        }

        var hashedPassword = CryptoJS.SHA256(password).toString();
        //var salt = CryptoJS.SHA256(salt_value).toString();

        $.post('#', { action: 'login', uid: username, passLogIn: hashedPassword, salt: salt_value }, function (data) {
            if (data.ok === 1) {
                loginAttempts = 0;
                $(location).attr('href', 'index.html');
                alert("Chào mừng bạn!");
                checklogin(username)
            } else {
                loginAttempts++;
                if (loginAttempts >= 3) {
                    $("#captchaSection").show();
                    $("#captchaImage").attr("src", "api.aspx?action=generate_captcha");
                }
                alert("Đăng nhập thất bại: " + data.msg);
                loginAttempts++;
            }
        }, 'json');
    }
    checklogin();

    function displayUserInfo(data) {
        $('#username').val(data.username);
        $('#gmail').val(data.gmail);
        $('#dienthoai').val(data.dienthoai);
        $('#name').val(data.name);
        $('#maso').val(data.maso);
        $('#diachi').val(data.diachi);
    }

    $('#update_infor').click(function () {
        var html =
            `<form id="addDeviceForm">
                         <div class="mb-3">
                            <label for="passLogIn" class="form-label">Nhập lại mật khẩu</label>
                            <input type="password" class="form-control" id="passLogIn" placeholder="Nhập mã phòng">
                        </div>
                    </form>`;

        var confirmBox = $.confirm({
            title: 'Sửa thông tin',
            content: html,
            boxWidth: '30%',
            theme: 'material',
            buttons: {
                save: {
                    text: 'Xác nhận',
                    btnClass: 'btn-primary',
                    action: function () {
                        var passLogIn = $("#passLogIn").val().trim();

                        var uid = $("#username").val().trim();
                        var passnew = $("#password").val().trim();
                        var gmail = $("#gmail").val().trim();
                        var dienthoai = $("#dienthoai").val().trim();
                        var name = $("#name").val().trim();
                        var maso = $("#maso").val().trim();
                        var diachi = $("#diachi").val().trim();
                        if (!uid || !passnew || !gmail || !dienthoai || !name || !maso || !diachi) {
                            $.alert('Vui lòng nhập đầy đủ thông tin!');
                            return false;
                        }

                        $.post('#', {
                            action: 'change_tt',
                            uid: uid,
                            passLogIn: CryptoJS.SHA256(passLogIn).toString(),
                            passnew: CryptoJS.SHA256(passnew).toString(),
                            name: name,
                            maso: maso,
                            gmail: gmail,
                            dienthoai: dienthoai,
                            diachi: diachi 
                        }, function (data) {
                            if (data.ok) {
                                $.alert('Đã Sửa Thành Công ! Vui lòng đăng nhập lại !');
                                confirmBox.close();
                                $(location).attr('href', 'dangnhap.html');

                            } else {
                                $.alert('Mật khẩu sai rồi !');
                            }
                        }, 'json');
                        return false;
                    }
                },
                cancel: {
                    text: 'Hủy',
                    btnClass: 'btn-secondary',
                }
            }
        });
    
    });

    function checklogin() {
        $.post('#', { action: 'check_login'}, function (data) {
            if (data.ok === 1) {
                loadTable(function () {
                    control(data.level); 
                });
                $('.nav-link .text-gray-600').text(data.name);
                displayUserInfo(data); 
            } else {
                alert("Đăng nhập để vào web xem nhé");
            }
        }, 'json');
    }
    window.xuat_excel_phong = function() {
        const rows = [];
        document.querySelectorAll(".phong").forEach((phong) => {
            const maPhong = phong.querySelector(".tt_phong h4:nth-child(2)").innerText.split(": ")[1];
            const tenPhong = phong.querySelector(".tt_phong h4:nth-child(3)").innerText.split(": ")[1];
            const trangThai = phong.querySelector(".tt_phong h4:nth-child(4)").innerText.split(": ")[1];
            rows.push({ "Mã Phòng": maPhong, "Tên Phòng": tenPhong, "Trạng Thái": trangThai });
        });

        const worksheet = XLSX.utils.json_to_sheet(rows);
        const workbook = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(workbook, worksheet, "Danh sách phòng");
        XLSX.writeFile(workbook, "DanhSachPhong.xlsx");
    }

    window.xuat_excel_thiet_bi = function () {
        const rows = [];
        document.querySelectorAll(".phong").forEach((phong) => {
            const maPhong = phong.querySelector(".tt_phong h4:nth-child(2)").innerText.split(": ")[1];
            const tenPhong = phong.querySelector(".tt_phong h4:nth-child(3)").innerText.split(": ")[1];
            phong.querySelectorAll(".thietbi").forEach((thietbi) => {
                const ttThietBi = thietbi.querySelector(".tt_thietbi").innerText.split(", ");
                const maThietBi = ttThietBi[0].split(": ")[1];
                const tenThietBi = ttThietBi[1].split(": ")[1];
                const trangThaiTB = ttThietBi[2].split(": ")[1];
                rows.push({
                    "Mã Phòng": maPhong,
                    "Tên Phòng": tenPhong,
                    "Mã Thiết Bị": maThietBi,
                    "Tên Thiết Bị": tenThietBi,
                    "Trạng Thái Thiết Bị": trangThaiTB
                });
            });
        });

        const worksheet = XLSX.utils.json_to_sheet(rows);
        const workbook = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(workbook, worksheet, "Danh sách thiết bị");
        XLSX.writeFile(workbook, "DanhSachThietBi.xlsx");
    }


    function showCaptcha() {
        $("#captchaSection").show();
        refreshCaptcha();
    }

    function validateCaptcha(callback) {
        var captchaInput = $("#captchaInput").val();
        $.post('#', { action: 'validate_captcha', captcha: captchaInput  }, function (data) {
            if (data.ok) {
                loginAttempts = 0;
                callback();
            } else {
                alert("CAPTCHA sai, vui lòng nhập lại.");
                refreshCaptcha();
            }
        }, 'json');
    }

    function login() {
        if (loginAttempts >= 3) {
            validateCaptcha(function () {
                get_salt(function (salt_value) {
                    performLogin(salt_value);
                });
            });
        } else {
            get_salt(function (salt_value) {
                performLogin(salt_value);
            });
        }
        
    }

    $("#dangky").click(function (event) {
        event.preventDefault(); 
        dangky();
    });
    function refreshCaptcha() {
        $("#captchaImage").attr("src", "api.aspx?action=generate_captcha&" + new Date().getTime());
    }

    $("#refreshCaptcha").click(refreshCaptcha);

    function control(level) {
        level = parseInt(level);
        console.log($('.change').length, $('.button-add').length, $('.tb_nut').length);
        switch (level) {
            case 1:
                $('.change, .button-add, .tb_nut').hide();
                $('.change #edit-btn, .change #delete-btn, .tb_nut #edit-btn, .tb_nut #delete-btn').hide();
                break;
            case 2:
                $('.change #add-btn, .button-add').show();
                $('.change #edit-btn, .change #delete-btn, .tb_nut #edit-btn, .tb_nut #delete-btn').hide();
                break;
            case 3:
                $('.change, .button-add, .tb_nut').show();
                $('.change #edit-btn, .change #delete-btn, .tb_nut #edit-btn, .tb_nut #delete-btn').show();
                break;
            default:
                console.log("Trạng thái không rõ");
        }
    }

    $("#dangnhap").on("click", function () {
        login();
    });

    window.loadtimkiemphong = function () {
        var timkiem = $("#timkieminput").val().trim();
        var container = $(".data_phong_tb");
        container.empty();
        container.append('<div>Đang tải dữ liệu...</div>');

        $.post('#', { action: 'search', timkiem: timkiem }, function (data) {
            container.empty();
            if (data.length > 0) {
                data.forEach(function (phong) {
                    var phongDiv = $("<div class='phong'></div>");
                    phongDiv.append($("<div></div>").html(
                        `<div class="tt_phong">
                            <h3 style="text-align: center;"> Thông Tin Phòng Học </h3>
                            <h4>Mã Phòng Học: ${phong.maphong}</h4>
                            <h4>Tên Phòng Học: ${phong.tenphong}</h4>
                            <h4>Trạng Thái Phòng Học : ${statusPhong(phong.ttphong)}</h4>
                        </div>
                        <div class="change">
                            <button class="button_phong" id="add-btn" onclick="them_phong()">Thêm phòng</button>
                            <button class="button_phong" id="edit-btn" onclick="sua_phong('${phong.maphong}', this)">Sửa</button>
                            <button class="button_phong" id="delete-btn" onclick="xoa_phong('${phong.maphong}', this)">Xóa</button>
                        </div>
                    `));

                    var thietBiListDiv = $("<div class='thiet_bi_list'></div>");
                    thietBiListDiv.append('<strong style="display: flex; justify-content: space-around;">Danh Sách Thiết Bị:</strong>');
                    thietBiListDiv.append($("<div class='button-add'></div>").html(`<button class="button_tb" id="add-btn" onclick="them_tb('${phong.maphong}', this)">Thêm thiết bị</button>`));

                    if (phong.thiet_bi_list && phong.thiet_bi_list.length > 0) {
                        phong.thiet_bi_list.forEach(function (thietBi) {
                            var thietBiDiv = $("<div class='thietbi'></div>");
                            thietBiDiv.append($('<div class="tt_thietbi"></div>').text("Mã Thiết Bị: " + thietBi.matb + ", Tên: " + thietBi.tentb + ", Trạng Thái: " + status(thietBi.tttbi)));
                            thietBiDiv.append($('<div class="tb_nut"></div>').html(
                                `<button class="button_tb" id="edit-btn" onclick="sua_tb('${thietBi.matb}', this)">Sửa</button>
                                     <button class="button_tb" id="delete-btn" onclick="xoa_tb('${thietBi.matb}', this)">Xóa</button>`
                            ));
                            thietBiListDiv.append(thietBiDiv);
                        });
                    } else {
                        thietBiListDiv.append("<div>Không có thiết bị nào trong phòng này.</div>");
                    }

                    phongDiv.append(thietBiListDiv);
                    container.append(phongDiv);
                });
            } else {
                container.append('<div>Không có dữ liệu nào.</div>');
            }
            if (callback) callback();
        }, 'json');
    }        
 
    $("#timkiemphong").on("click", function () {
        loadtimkiemphong();
    });
});
